(** Resonance — the pure architecture with learnable frequencies.

    tokens |> strike |> resonate |> listen |> softmax

    Learning:
      drive signatures — gradient descent on dot product error
      oscillator ω, γ  — analytical dH/dω retuning

    The bank IS the model. The drive IS the readout.
    Learning IS retuning the instrument AND adjusting the ear.
    No matrix multiply. No transform. Just physics and dot products. *)

let vocab_size = 256

(* --- Probability --- *)

let softmax v =
  let mx = Array.fold_left Float.max neg_infinity v in
  let exps = Array.map (fun l -> exp (l -. mx)) v in
  let sum = Array.fold_left ( +. ) 0.0 exps in
  Array.map (fun e -> e /. sum) exps

let cross_entropy ~target probs =
  -. log (Float.max 1e-10 probs.(target))

let logit_gradient ~target probs =
  Array.mapi (fun i p -> p -. (if i = target then 1.0 else 0.0)) probs

let sample ~temperature logits =
  let probs = Array.map (fun l -> l /. temperature) logits |> softmax in
  let r = Random.float 1.0 in
  let i = ref 0 and acc = ref 0.0 in
  while !acc < r && !i < Array.length probs - 1 do
    acc := !acc +. probs.(!i); if !acc < r then incr i done; !i

(* --- Model --- *)

type t = {
  mutable oscillators : Oscillator.t array;
  drive : float array array;       (** 256 × state_dim: full-dimensional *)
  mutable kernels : Bank.kernels;
  n_osc : int;
  state_dim : int;
  seq_len : int;
}

let create n_osc seq_len =
  let state_dim = 2 * n_osc in
  let scale = 1.0 /. sqrt (Float.of_int state_dim) in
  let oscillators = Oscillator.spread n_osc in
  {
    oscillators;
    drive = Array.init vocab_size (fun _ ->
      Array.init state_dim (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc; state_dim; seq_len;
  }

(* --- Pipeline --- *)

(** Strike: first n_osc components of drive signature *)
let strike model token =
  Array.init model.n_osc (fun i -> model.drive.(token).(i))

(** Resonate: FFT convolve drives with h(t) *)
let resonate model tokens =
  let drives = Array.map (strike model) tokens in
  Bank.encode model.oscillators model.kernels drives

(** Listen: full-dimensional dot product *)
let listen model state =
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to model.state_dim - 1 do
      acc := !acc +. state.(i) *. sig_.(i)
    done; !acc
  ) model.drive

(** Forward: state → logits *)
let forward model state = listen model state

(** Predict: state → probabilities *)
let predict model state = forward model state |> softmax

(* --- Learning --- *)

(** Update drive signatures *)
let update_drives model state d_logits ~lr =
  Array.iteri (fun b sig_ ->
    let dl = d_logits.(b) in
    if Float.abs dl > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. lr *. dl *. state.(j)
      ) sig_
  ) model.drive

(** Compute d_loss/d_state from logit gradient *)
let state_gradient model d_logits =
  Array.init model.state_dim (fun j ->
    let acc = ref 0.0 in
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-8 then
        acc := !acc +. dl *. sig_.(j)
    ) model.drive; !acc)

(** Retune oscillator frequencies from prediction error.
    d_loss/d_ω_k = Σ_t d_loss/d_state_t · d_state_t/d_ω_k
    where d_state/d_ω is computed by convolving drives with dh/dω *)
let retune_frequencies model tokens d_states ~lr =
  let drives = Array.map (strike model) tokens in
  let d_omega = Bank.encode_deriv model.oscillators model.kernels drives in
  let n_osc = model.n_osc in
  let seq_len = Array.length tokens in
  (* For each oscillator, accumulate gradient across positions *)
  let new_oscs = Array.mapi (fun k osc ->
    let grad = ref 0.0 in
    for t = 0 to seq_len - 2 do
      let d_state = d_states.(t) in
      let dpos, dvel = d_omega.(k).(t) in
      (* d_loss/d_ω_k += d_state_pos_k × dh_pos/dω + d_state_vel_k × dh_vel/dω *)
      grad := !grad +. d_state.(k) *. dpos +. d_state.(n_osc + k) *. dvel
    done;
    let new_omega = Float.max 0.01 (osc.Oscillator.omega0 -. lr *. !grad) in
    Oscillator.create new_omega osc.Oscillator.gamma
  ) model.oscillators in
  model.oscillators <- new_oscs;
  model.kernels <- Bank.precompute_kernels new_oscs model.seq_len

(** Train on a token sequence *)
let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let lr_scaled = lr /. Float.of_int seq_len in
  let states = resonate model tokens in

  (* Forward + gradient at each position *)
  let total_loss = ref 0.0 in
  let d_states = Array.init (seq_len - 1) (fun t ->
    let probs = predict model states.(t) in
    let target = tokens.(t + 1) in
    total_loss := !total_loss +. cross_entropy ~target probs;
    let d_logits = logit_gradient ~target probs in
    update_drives model states.(t) d_logits ~lr:lr_scaled;
    state_gradient model d_logits
  ) in

  (* Retune oscillator frequencies every step *)
  retune_frequencies model tokens d_states ~lr:(lr_scaled *. 0.1);

  !total_loss /. Float.of_int (seq_len - 1)

(* --- Generation --- *)

let generate model seed ~n_gen ~temperature =
  let buf = Buffer.create n_gen in
  let context = Array.copy seed in
  let n = Array.length context in
  for _ = 1 to n_gen do
    let states = resonate model context in
    let logits = forward model states.(n - 1) in
    let next = sample ~temperature logits in
    Buffer.add_char buf
      (if next >= 32 && next < 127 then Char.chr next else '.');
    Array.blit context 1 context 0 (n - 1);
    context.(n - 1) <- next
  done;
  Buffer.contents buf

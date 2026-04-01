(** Resonance — the simplest possible architecture.

    tokens |> strike |> resonate |> listen |> softmax

    Four functions. No transform. No prism. No matrix multiply.

    The bank IS the model — FFT convolution decomposes history.
    The drive IS the readout — full-dimensional matched filter.
    Learning IS retuning — adjust frequencies + drive signatures.

    Each drive signature is 2×n_osc: separate weights for position
    and velocity. Phase information preserved — an oscillator at
    peak amplitude means something different than one crossing zero. *)

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
  oscillators : Oscillator.t array;
  drive : float array array;       (** 256 × state_dim: full-dimensional signatures *)
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

(* --- The pipeline --- *)

(** Strike: first n_osc components of drive are the force *)
let strike model token =
  Array.init model.n_osc (fun i -> model.drive.(token).(i))

(** Resonate: FFT convolve with oscillator impulse responses *)
let resonate model tokens =
  let drives = Array.map (strike model) tokens in
  Bank.encode model.oscillators model.kernels drives

(** Listen: dot product of full state with full drive signature *)
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

(** Backprop through listen: drive^T × d_logits → d_state *)
let listen_backward model d_logits =
  Array.init model.state_dim (fun j ->
    let acc = ref 0.0 in
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-8 then
        acc := !acc +. dl *. sig_.(j)
    ) model.drive; !acc)

(** Update drive signatures from output gradient *)
let update_drives model state d_logits ~lr =
  Array.iteri (fun b sig_ ->
    let dl = d_logits.(b) in
    if Float.abs dl > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. lr *. dl *. state.(j)
      ) sig_
  ) model.drive

(** Learn from one position *)
let learn_position model ~state ~target ~lr =
  let probs = predict model state in
  let loss = cross_entropy ~target probs in
  let d_logits = logit_gradient ~target probs in
  update_drives model state d_logits ~lr;
  loss

(* --- Training --- *)

(** Train on a token sequence *)
let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let lr_scaled = lr /. Float.of_int seq_len in
  let states = resonate model tokens in
  let total_loss = ref 0.0 in
  for t = 0 to seq_len - 2 do
    total_loss := !total_loss +.
      learn_position model ~state:states.(t) ~target:tokens.(t + 1) ~lr:lr_scaled
  done;
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

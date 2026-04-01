(** Resonance — the proven architecture.

    tokens |> strike |> resonate |> transform |> listen |> softmax

    Five functions. Two learned components:
      drive — full-dimensional signatures for strike and listen
      W     — 192 dot products that rotate physics basis to prediction basis

    The bank decomposes history through oscillator physics.
    W rotates the state so drive signatures can detect patterns.
    Drive signatures match the rotated state to predict next byte.

    All dot products. All parallel. All CPU. *)

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
  drive : float array array;       (** 256 × state_dim: strike and listen *)
  w : float array;                 (** state_dim × state_dim: rotation *)
  mutable kernels : Bank.kernels;
  n_osc : int;
  state_dim : int;
  seq_len : int;
}

let create n_osc seq_len =
  let state_dim = 2 * n_osc in
  let scale = 1.0 /. sqrt (Float.of_int state_dim) in
  let rand () = (Random.float 2.0 -. 1.0) *. scale in
  let oscillators = Oscillator.spread n_osc in
  {
    oscillators;
    drive = Array.init vocab_size (fun _ ->
      Array.init state_dim (fun _ -> rand ()));
    (* W starts as identity + small noise: model begins at pure architecture baseline *)
    w = Array.init (state_dim * state_dim) (fun k ->
      let i = k / state_dim and j = k mod state_dim in
      (if i = j then 1.0 else 0.0) +. rand () *. 0.01);
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc; state_dim; seq_len;
  }

(* --- Pipeline --- *)

(** Strike: first n_osc components of drive as forces *)
let strike model token =
  Array.init model.n_osc (fun i -> model.drive.(token).(i))

(** Resonate: FFT convolve drives with h(t) *)
let resonate model tokens =
  let drives = Array.map (strike model) tokens in
  Bank.encode model.oscillators model.kernels drives

(** Transform: W rotates state from physics basis to prediction basis *)
let transform model state =
  let dim = model.state_dim in
  Array.init dim (fun i ->
    let acc = ref 0.0 in
    let base = i * dim in
    for j = 0 to dim - 1 do
      acc := !acc +. model.w.(base + j) *. state.(j)
    done; !acc)

(** Listen: full-dimensional dot product with each drive signature *)
let listen model transformed =
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to model.state_dim - 1 do
      acc := !acc +. transformed.(i) *. sig_.(i)
    done; !acc
  ) model.drive

(** Forward: state → logits *)
let forward model state =
  state |> transform model |> listen model

(** Predict: state → probabilities *)
let predict model state =
  forward model state |> softmax

(* --- Learning --- *)

(** Gradient through listen: d_transformed = drive^T × d_logits *)
let listen_backward model d_logits =
  Array.init model.state_dim (fun j ->
    let acc = ref 0.0 in
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-8 then
        acc := !acc +. dl *. sig_.(j)
    ) model.drive; !acc)

(** Update W: gradient descent *)
let update_w model state d_transformed ~lr =
  let dim = model.state_dim in
  Array.iteri (fun i di ->
    if Float.abs di > 1e-8 then begin
      let base = i * dim in
      Array.iteri (fun j sj ->
        model.w.(base + j) <- model.w.(base + j) -. lr *. di *. sj
      ) state end
  ) d_transformed

(** Update drives: gradient descent *)
let update_drives model transformed d_logits ~lr =
  Array.iteri (fun b sig_ ->
    let dl = d_logits.(b) in
    if Float.abs dl > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. lr *. dl *. transformed.(j)
      ) sig_
  ) model.drive

(* --- Training --- *)

(** Train: accumulate W gradient across positions, update once *)
let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let dim = model.state_dim in
  let lr_scaled = lr /. Float.of_int seq_len in
  let states = resonate model tokens in

  let w_grad = Array.make (dim * dim) 0.0 in
  let total_loss = ref 0.0 in

  for t = 0 to seq_len - 2 do
    let state = states.(t) in
    let target = tokens.(t + 1) in
    let transformed = transform model state in
    let probs = predict model state in
    total_loss := !total_loss +. cross_entropy ~target probs;

    let d_logits = logit_gradient ~target probs in
    let d_transformed = listen_backward model d_logits in

    (* Accumulate W gradient *)
    Array.iteri (fun i di ->
      if Float.abs di > 1e-8 then begin
        let base = i * dim in
        Array.iteri (fun j sj ->
          w_grad.(base + j) <- w_grad.(base + j) +. di *. sj
        ) state end
    ) d_transformed;

    (* Drive updates are per-byte, apply immediately *)
    update_drives model transformed d_logits ~lr:lr_scaled
  done;

  (* Single W update *)
  Array.iteri (fun i g ->
    model.w.(i) <- model.w.(i) -. lr_scaled *. g
  ) w_grad;

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

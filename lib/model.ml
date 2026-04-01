(** Text model — the pipeline.

    tokens |> strike |> resonate |> transform |> listen |> predict

    Three components:
      drive — byte → oscillator force (strike) and comparison (listen)
      W     — learned projection from bank state to prediction space
      bank  — oscillator impulse responses, FFT convolved (physics) *)

let vocab_size = 256

(* --- Linear algebra primitives --- *)

(** Flat row-major matrix × vector *)
let mat_vec w ~rows ~cols x =
  Array.init rows (fun i ->
    let acc = ref 0.0 in
    let base = i * cols in
    for j = 0 to cols - 1 do
      acc := !acc +. w.(base + j) *. x.(j)
    done; !acc)

(** Sparse outer product update: W += scale × a ⊗ b (skips small a_i) *)
let outer_update w ~cols a b ~scale =
  Array.iteri (fun i ai ->
    if Float.abs ai > 1e-8 then
      let base = i * cols in
      Array.iteri (fun j bj ->
        w.(base + j) <- w.(base + j) +. scale *. ai *. bj
      ) b
  ) a

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
    acc := !acc +. probs.(!i); if !acc < r then incr i
  done; !i

(* --- Model --- *)

type t = {
  oscillators : Oscillator.t array;
  drive : float array array;
  w : float array;
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
    drive = Array.init vocab_size (fun _ -> Array.init n_osc (fun _ -> rand ()));
    w = Array.init (state_dim * state_dim) (fun _ -> rand ());
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc; state_dim; seq_len;
  }

(* --- Pipeline: strike → resonate → transform → listen → predict --- *)

(** Strike: byte selects a drive signature *)
let strike model token = model.drive.(token)

(** Resonate: FFT convolve drives with oscillator impulse responses *)
let resonate model tokens =
  let drives = Array.map (strike model) tokens in
  Bank.encode model.oscillators model.kernels drives

(** Transform: project bank state through learned W *)
let transform model state =
  mat_vec model.w ~rows:model.state_dim ~cols:model.state_dim state

(** Listen: dot product of [pos + vel] with each drive signature *)
let listen model transformed =
  let n = model.n_osc in
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to n - 1 do
      acc := !acc +. sig_.(i) *. (transformed.(i) +. transformed.(n + i))
    done; !acc
  ) model.drive

(** Forward: state → logits *)
let forward model state =
  state |> transform model |> listen model

(** Predict: state → probability distribution *)
let predict model state =
  state |> forward model |> softmax

(* --- Gradient computation --- *)

(** Backprop through listen: drive^T × d_logits → d_transformed.
    Sparse: only bytes with significant gradient contribute. *)
let listen_backward model d_logits =
  let dim = model.state_dim and n = model.n_osc in
  Array.init dim (fun j ->
    let acc = ref 0.0 in
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-8 then
        acc := !acc +. dl *. sig_.(j mod n)
    ) model.drive;
    !acc)

(** Update W: gradient descent on W from one position's error *)
let update_w model state d_transformed ~lr =
  outer_update model.w ~cols:model.state_dim d_transformed state ~scale:(-.lr)

(** Update drives: adjust each byte's signature from logit gradient *)
let update_drives model transformed d_logits ~lr =
  let n = model.n_osc in
  Array.iteri (fun b sig_ ->
    let dl = d_logits.(b) in
    if Float.abs dl > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. lr *. dl *. (transformed.(j) +. transformed.(n + j))
      ) sig_
  ) model.drive

(** Learn from one position: forward → loss → backward → update *)
let learn_position model ~state ~target ~lr =
  let transformed = transform model state in
  let probs = predict model state in
  let loss = cross_entropy ~target probs in
  let d_logits = logit_gradient ~target probs in
  let d_transformed = listen_backward model d_logits in
  update_w model state d_transformed ~lr;
  update_drives model transformed d_logits ~lr;
  loss

(* --- Training --- *)

(** Train on a token sequence: resonate → learn at each position *)
let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let lr_scaled = lr /. Float.of_int seq_len in
  let states = resonate model tokens in
  let total_loss = ref 0.0 in
  for t = 0 to seq_len - 2 do
    let loss = learn_position model
      ~state:states.(t) ~target:tokens.(t + 1) ~lr:lr_scaled in
    total_loss := !total_loss +. loss
  done;
  !total_loss /. Float.of_int (seq_len - 1)

(* --- Generation --- *)

(** Shift context window: drop oldest, append new *)
let shift_context context next =
  let n = Array.length context in
  Array.blit context 1 context 0 (n - 1);
  context.(n - 1) <- next

(** Generate text from a seed *)
let generate model seed ~n_gen ~temperature =
  let context = Array.copy seed in
  let buf = Buffer.create n_gen in
  for _ = 1 to n_gen do
    let states = resonate model context in
    let logits = forward model states.(Array.length context - 1) in
    let next = sample ~temperature logits in
    Buffer.add_char buf
      (if next >= 32 && next < 127 then Char.chr next else '.');
    shift_context context next
  done;
  Buffer.contents buf

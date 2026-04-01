(** Text model — the pipeline.

    tokens |> strike |> resonate |> transform |> listen |> predict

    Three learned components:
      drive   — byte → oscillator force (strike and listen)
      W       — state → prediction space (learned transformation)
      bank    — oscillator impulse responses (physics, fixed) *)

let vocab_size = 256

(* --- Vector ops as named functions --- *)

let dot a b =
  let n = Array.length a in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do acc := !acc +. a.(i) *. b.(i) done;
  !acc

let scale_add ~scale src dst =
  Array.iteri (fun i s -> dst.(i) <- dst.(i) +. scale *. s) src

let outer_add ~scale a b dst ~cols =
  Array.iteri (fun i ai ->
    if Float.abs ai > 1e-8 then
      let base = i * cols in
      Array.iteri (fun j bj ->
        dst.(base + j) <- dst.(base + j) +. scale *. ai *. bj) b
  ) a

(* --- Probability --- *)

let softmax v =
  let mx = Array.fold_left Float.max neg_infinity v in
  let exps = Array.map (fun l -> exp (l -. mx)) v in
  let sum = Array.fold_left ( +. ) 0.0 exps in
  Array.map (fun e -> e /. sum) exps

let cross_entropy ~target probs =
  -. log (Float.max 1e-10 probs.(target))

let output_error ~target probs =
  Array.mapi (fun i p -> p -. (if i = target then 1.0 else 0.0)) probs

let sample probs ~temperature =
  let scaled = Array.map (fun l -> l /. temperature) probs |> softmax in
  let r = Random.float 1.0 in
  let i = ref 0 in
  let acc = ref 0.0 in
  while !acc < r && !i < Array.length scaled - 1 do
    acc := !acc +. scaled.(!i);
    if !acc < r then incr i
  done;
  !i

(* --- Model --- *)

type t = {
  oscillators : Oscillator.t array;
  drive : float array array;       (** strike and listen signatures *)
  w : float array;                 (** flat row-major transform matrix *)
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

(** Strike: byte → drive forces *)
let strike model token = model.drive.(token)

(** Resonate: drives → FFT convolve with h(t) → causal states *)
let resonate model drives =
  Bank.encode model.oscillators model.kernels drives

(** Transform: state × W → prediction space *)
let transform model state =
  let dim = model.state_dim in
  Array.init dim (fun i ->
    let acc = ref 0.0 in
    let base = i * dim in
    for j = 0 to dim - 1 do
      acc := !acc +. model.w.(base + j) *. state.(j)
    done;
    !acc)

(** Listen: transformed state → logit per byte via drive dot product *)
let listen model transformed =
  let n = model.n_osc in
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to n - 1 do
      acc := !acc +. sig_.(i) *. (transformed.(i) +. transformed.(n + i))
    done;
    !acc
  ) model.drive

(** Forward: state → logits (the inference pipeline) *)
let forward model state =
  state |> transform model |> listen model

(** Predict: state → probability distribution *)
let predict model state =
  state |> forward model |> softmax

(* --- Gradient computation (per position, pure) --- *)

(** Gradient of the transform output w.r.t. output error.
    Sparse: only target + significant bytes contribute. *)
let transform_gradient model ~target d_logits =
  let dim = model.state_dim in
  let n = model.n_osc in
  let grad = Array.make dim 0.0 in
  (* Target contribution (always significant) *)
  let target_dl = d_logits.(target) in
  let target_sig = model.drive.(target) in
  for j = 0 to dim - 1 do
    grad.(j) <- target_dl *. target_sig.(j mod n)
  done;
  (* Non-target contributions (skip small ones) *)
  Array.iteri (fun b sig_ ->
    if b <> target then begin
      let dl = d_logits.(b) in
      if Float.abs dl > 0.01 then
        for j = 0 to dim - 1 do
          grad.(j) <- grad.(j) +. dl *. sig_.(j mod n)
        done
    end
  ) model.drive;
  grad

(** Per-position gradient: pure function, no shared state *)
let position_gradient model state target =
  let transformed = transform model state in
  let probs = softmax (listen model transformed) in
  let loss = cross_entropy ~target probs in
  let d_logits = output_error ~target probs in
  let d_transform = transform_gradient model ~target d_logits in
  (loss, d_transform, d_logits, transformed)

(* --- Training --- *)

(** Accumulate W gradient from one position *)
let accumulate_w_grad state d_transform w_grad ~dim =
  for i = 0 to dim - 1 do
    let di = d_transform.(i) in
    if Float.abs di > 1e-8 then begin
      let base = i * dim in
      for j = 0 to dim - 1 do
        w_grad.(base + j) <- w_grad.(base + j) +. di *. state.(j)
      done
    end
  done

(** Accumulate drive gradient from one position *)
let accumulate_drive_grad model d_logits transformed drive_grad =
  let n = model.n_osc in
  Array.iteri (fun b dg ->
    let dl = d_logits.(b) in
    if Float.abs dl > 1e-6 then
      for j = 0 to n - 1 do
        dg.(j) <- dg.(j) +. dl *. (transformed.(j) +. transformed.(n + j))
      done
  ) drive_grad

(** Apply accumulated gradients to model weights *)
let apply_gradients model w_grad drive_grad ~scale =
  Array.iteri (fun i g -> model.w.(i) <- model.w.(i) -. scale *. g) w_grad;
  Array.iteri (fun b dg ->
    let sig_ = model.drive.(b) in
    Array.iteri (fun j g -> sig_.(j) <- sig_.(j) -. scale *. g) dg
  ) drive_grad

(** Train on a sequence: strike → resonate → parallel gradients → update *)
let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let dim = model.state_dim in

  (* Forward: the physics pipeline *)
  let drives = Array.map (strike model) tokens in
  let states = resonate model drives in

  (* Gradient: parallel across positions *)
  let grads = Par.init (seq_len - 1) (fun t ->
    position_gradient model states.(t) tokens.(t + 1)) in

  (* Reduce: accumulate gradients from all positions *)
  let w_grad = Array.make (dim * dim) 0.0 in
  let drive_grad = Array.init vocab_size (fun _ -> Array.make model.n_osc 0.0) in
  let total_loss = ref 0.0 in
  Array.iteri (fun t (loss, d_trans, d_logits, transformed) ->
    total_loss := !total_loss +. loss;
    accumulate_w_grad states.(t) d_trans w_grad ~dim;
    accumulate_drive_grad model d_logits transformed drive_grad
  ) grads;
  let total_loss = !total_loss in

  (* Update *)
  apply_gradients model w_grad drive_grad ~scale:(lr /. Float.of_int seq_len);

  total_loss /. Float.of_int (seq_len - 1)

(* --- Generation --- *)

(** Generate: strike → resonate → predict → sample → repeat *)
let generate model seed ~n_gen ~temperature =
  let buf = Buffer.create n_gen in
  let context = Array.copy seed in
  let n = Array.length context in
  for _ = 1 to n_gen do
    let drives = Array.map (strike model) context in
    let states = resonate model drives in
    let probs = predict model states.(n - 1) in
    let next = sample probs ~temperature in
    Buffer.add_char buf
      (if next >= 32 && next < 127 then Char.chr next else '.');
    Array.blit context 1 context 0 (n - 1);
    context.(n - 1) <- next
  done;
  Buffer.contents buf

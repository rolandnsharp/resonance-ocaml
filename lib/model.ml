(** Text model — the pipeline.

    tokens |> strike |> resonate |> transform |> listen |> predict

    Batch parallel: each CPU core processes a different sequence.
    Gradients averaged across the batch, applied once. *)

let vocab_size = 256

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
  let i = ref 0 and acc = ref 0.0 in
  while !acc < r && !i < Array.length scaled - 1 do
    acc := !acc +. scaled.(!i); if !acc < r then incr i done;
  !i

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
    done; !acc)

(** Listen: transformed state → logit per byte *)
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

(** Predict: state → probabilities *)
let predict model state =
  state |> forward model |> softmax

(* --- Gradient types --- *)

type grads = {
  w_grad : float array;
  drive_grad : float array array;
  mutable loss : float;
  mutable n_positions : int;
}

let zero_grads dim n_osc = {
  w_grad = Array.make (dim * dim) 0.0;
  drive_grad = Array.init vocab_size (fun _ -> Array.make n_osc 0.0);
  loss = 0.0;
  n_positions = 0;
}

(** Compute gradients for one full sequence. Pure — no mutation. *)
let sequence_gradients model tokens =
  let seq_len = Array.length tokens in
  let dim = model.state_dim in
  let n_osc = model.n_osc in

  let drives = Array.map (strike model) tokens in
  let states = resonate model drives in
  let g = zero_grads dim n_osc in

  for t = 0 to seq_len - 2 do
    let state = states.(t) in
    let target = tokens.(t + 1) in
    let transformed = transform model state in
    let probs = softmax (listen model transformed) in
    g.loss <- g.loss +. cross_entropy ~target probs;

    let d_logits = output_error ~target probs in

    (* Sparse transform gradient: target + significant bytes *)
    let d_trans = Array.make dim 0.0 in
    let target_dl = d_logits.(target) in
    let target_sig = model.drive.(target) in
    for j = 0 to dim - 1 do
      d_trans.(j) <- target_dl *. target_sig.(j mod n_osc)
    done;
    for b = 0 to vocab_size - 1 do
      if b <> target then begin
        let dl = d_logits.(b) in
        if Float.abs dl > 0.01 then begin
          let sig_ = model.drive.(b) in
          for j = 0 to dim - 1 do
            d_trans.(j) <- d_trans.(j) +. dl *. sig_.(j mod n_osc)
          done end end
    done;

    (* Accumulate W gradient *)
    for i = 0 to dim - 1 do
      let di = d_trans.(i) in
      if Float.abs di > 1e-8 then begin
        let base = i * dim in
        for j = 0 to dim - 1 do
          g.w_grad.(base + j) <- g.w_grad.(base + j) +. di *. state.(j)
        done end
    done;

    (* Accumulate drive gradient *)
    for b = 0 to vocab_size - 1 do
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-6 then begin
        let dg = g.drive_grad.(b) in
        for j = 0 to n_osc - 1 do
          dg.(j) <- dg.(j) +. dl *. (transformed.(j) +. transformed.(n_osc + j))
        done end
    done
  done;
  g.n_positions <- seq_len - 1;
  g

(** Merge gradients from multiple sequences *)
let merge_grads grads_list dim =
  let total = zero_grads dim 0 in  (* drive_grad merged separately *)
  let first = List.hd grads_list in
  let n_osc = Array.length first.drive_grad.(0) in
  let merged_drive = Array.init vocab_size (fun _ -> Array.make n_osc 0.0) in
  let total_loss = ref 0.0 in
  let total_pos = ref 0 in
  List.iter (fun g ->
    total_loss := !total_loss +. g.loss;
    total_pos := !total_pos + g.n_positions;
    Array.iteri (fun i v ->
      total.w_grad.(i) <- total.w_grad.(i) +. v) g.w_grad;
    Array.iteri (fun b dg ->
      let m = merged_drive.(b) in
      Array.iteri (fun j v -> m.(j) <- m.(j) +. v) dg
    ) g.drive_grad
  ) grads_list;
  (!total_loss, !total_pos, total.w_grad, merged_drive)

(** Apply gradients to model *)
let apply_grads model w_grad drive_grad ~scale =
  Array.iteri (fun i g -> model.w.(i) <- model.w.(i) -. scale *. g) w_grad;
  Array.iteri (fun b dg ->
    let sig_ = model.drive.(b) in
    Array.iteri (fun j g -> sig_.(j) <- sig_.(j) -. scale *. g) dg
  ) drive_grad

(** Train: batch-parallel across CPU cores.
    Each core processes a different sequence. Gradients merged and applied once. *)
let train_batch model token_seqs ~lr =
  (* Each core computes gradients for one sequence *)
  let grads = Par.map (sequence_gradients model) token_seqs in
  (* Merge *)
  let total_loss, total_pos, w_grad, drive_grad =
    merge_grads (Array.to_list grads) model.state_dim in
  (* Apply *)
  let scale = lr /. Float.of_int total_pos in
  apply_grads model w_grad drive_grad ~scale;
  total_loss /. Float.of_int total_pos

(** Single sequence train (backward compat) *)
let train_sequence model tokens ~lr =
  train_batch model [| tokens |] ~lr

(* --- Generation --- *)

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

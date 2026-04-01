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
  w1 : float array;           (** state_dim → hidden_dim *)
  w2 : float array;           (** hidden_dim → state_dim *)
  mutable kernels : Bank.kernels;
  n_osc : int;
  state_dim : int;
  hidden_dim : int;
  seq_len : int;
}

let create n_osc seq_len =
  let state_dim = 2 * n_osc in
  let hidden_dim = state_dim * 2 in  (* 2x expansion like the Python FF *)
  let scale1 = 1.0 /. sqrt (Float.of_int state_dim) in
  let scale2 = 1.0 /. sqrt (Float.of_int hidden_dim) in
  let oscillators = Oscillator.spread n_osc in
  {
    oscillators;
    drive = Array.init vocab_size (fun _ ->
      Array.init n_osc (fun _ -> (Random.float 2.0 -. 1.0) *. scale1));
    w1 = Array.init (hidden_dim * state_dim) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale1);
    w2 = Array.init (state_dim * hidden_dim) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale2);
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc; state_dim; hidden_dim; seq_len;
  }

(** Strike: byte → drive forces *)
let strike model token = model.drive.(token)

(** Resonate: drives → FFT convolve with h(t) → causal states *)
let resonate model drives =
  Bank.encode model.oscillators model.kernels drives

(** SineGate: x × sin(x) — oscillation as activation *)
let sine_gate x = x *. sin x

(** Matrix-vector: y = W × x *)
let mat_vec w ~rows ~cols x =
  Array.init rows (fun i ->
    let acc = ref 0.0 in
    let base = i * cols in
    for j = 0 to cols - 1 do
      acc := !acc +. w.(base + j) *. x.(j)
    done; !acc)

(** Transposed matrix-vector: y = W^T × x *)
let mat_vec_t w ~rows ~cols x =
  Array.init cols (fun j ->
    let acc = ref 0.0 in
    for i = 0 to rows - 1 do
      acc := !acc +. w.(i * cols + j) *. x.(i)
    done; !acc)

(** Transform: W1 → SineGate → W2 *)
let transform model state =
  let hidden = mat_vec model.w1 ~rows:model.hidden_dim ~cols:model.state_dim state in
  let activated = Array.map sine_gate hidden in
  mat_vec model.w2 ~rows:model.state_dim ~cols:model.hidden_dim activated

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
  w1_grad : float array;
  w2_grad : float array;
  drive_grad : float array array;
  mutable loss : float;
  mutable n_positions : int;
}

let zero_grads model = {
  w1_grad = Array.make (model.hidden_dim * model.state_dim) 0.0;
  w2_grad = Array.make (model.state_dim * model.hidden_dim) 0.0;
  drive_grad = Array.init vocab_size (fun _ -> Array.make model.n_osc 0.0);
  loss = 0.0;
  n_positions = 0;
}

(** SineGate derivative: d(x sin x)/dx = sin x + x cos x *)
let sine_gate_deriv x = sin x +. x *. cos x

(** Compute gradients for one full sequence. *)
let sequence_gradients model tokens =
  let seq_len = Array.length tokens in
  let dim = model.state_dim in
  let hdim = model.hidden_dim in
  let n_osc = model.n_osc in

  let drives = Array.map (strike model) tokens in
  let states = resonate model drives in
  let g = zero_grads model in

  for t = 0 to seq_len - 2 do
    let state = states.(t) in
    let target = tokens.(t + 1) in

    (* Forward through W1 → SineGate → W2 *)
    let hidden = mat_vec model.w1 ~rows:hdim ~cols:dim state in
    let activated = Array.map sine_gate hidden in
    let transformed = mat_vec model.w2 ~rows:dim ~cols:hdim activated in

    let probs = softmax (listen model transformed) in
    g.loss <- g.loss +. cross_entropy ~target probs;

    let d_logits = output_error ~target probs in

    (* d_loss/d_transformed (sparse) *)
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

    (* Backprop through W2: d_activated = W2^T × d_trans *)
    let d_activated = mat_vec_t model.w2 ~rows:dim ~cols:hdim d_trans in

    (* W2 gradient: outer(d_trans, activated) *)
    for i = 0 to dim - 1 do
      let di = d_trans.(i) in
      if Float.abs di > 1e-8 then begin
        let base = i * hdim in
        for j = 0 to hdim - 1 do
          g.w2_grad.(base + j) <- g.w2_grad.(base + j) +. di *. activated.(j)
        done end
    done;

    (* Backprop through SineGate: d_hidden = d_activated * sine_gate'(hidden) *)
    let d_hidden = Array.init hdim (fun j ->
      d_activated.(j) *. sine_gate_deriv hidden.(j)) in

    (* W1 gradient: outer(d_hidden, state) *)
    for i = 0 to hdim - 1 do
      let di = d_hidden.(i) in
      if Float.abs di > 1e-8 then begin
        let base = i * dim in
        for j = 0 to dim - 1 do
          g.w1_grad.(base + j) <- g.w1_grad.(base + j) +. di *. state.(j)
        done end
    done;

    (* Drive gradient *)
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
let merge_grads model grads_list =
  let merged = zero_grads model in
  let total_loss = ref 0.0 in
  let total_pos = ref 0 in
  List.iter (fun g ->
    total_loss := !total_loss +. g.loss;
    total_pos := !total_pos + g.n_positions;
    Array.iteri (fun i v ->
      merged.w1_grad.(i) <- merged.w1_grad.(i) +. v) g.w1_grad;
    Array.iteri (fun i v ->
      merged.w2_grad.(i) <- merged.w2_grad.(i) +. v) g.w2_grad;
    Array.iteri (fun b dg ->
      let m = merged.drive_grad.(b) in
      Array.iteri (fun j v -> m.(j) <- m.(j) +. v) dg
    ) g.drive_grad
  ) grads_list;
  (!total_loss, !total_pos, merged)

(** Apply gradients to model *)
let apply_grads model grads ~scale =
  Array.iteri (fun i g -> model.w1.(i) <- model.w1.(i) -. scale *. g) grads.w1_grad;
  Array.iteri (fun i g -> model.w2.(i) <- model.w2.(i) -. scale *. g) grads.w2_grad;
  Array.iteri (fun b dg ->
    let sig_ = model.drive.(b) in
    Array.iteri (fun j g -> sig_.(j) <- sig_.(j) -. scale *. g) dg
  ) grads.drive_grad

(** Train: batch-parallel across CPU cores.
    Each core processes a different sequence. Gradients merged and applied once. *)
let train_batch model token_seqs ~lr =
  (* Each core computes gradients for one sequence *)
  let grads = Par.map (sequence_gradients model) token_seqs in
  let total_loss, total_pos, merged =
    merge_grads model (Array.to_list grads) in
  apply_grads model merged ~scale:(lr /. Float.of_int total_pos);
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

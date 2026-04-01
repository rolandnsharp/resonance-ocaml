(** Text model — FFT bank → learned projection → readout.

    tokens
    |> drive forces
    |> FFT convolve with h(t)    — oscillator physics
    |> W × state                 — learned transformation
    |> dot with drive signatures — weight-tied readout
    |> softmax → next byte

    The bank computes rich causal states (physics).
    W transforms them into prediction-useful space (learning).
    Drive signatures are used for both input and output (tying). *)

let vocab_size = 256

type t = {
  oscillators : Oscillator.t array;
  drive : float array array;   (** 256 × n_osc *)
  w : float array;             (** state_dim × state_dim, flat row-major *)
  mutable kernels : Bank.kernels;  (** precomputed FFT kernels *)
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
      Array.init n_osc (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    w = Array.init (state_dim * state_dim) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale);
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc;
    state_dim;
    seq_len;
  }

(** Matrix-vector multiply: y = W × x (W is flat row-major) *)
let mat_vec w ~rows ~cols x =
  Array.init rows (fun i ->
    let acc = ref 0.0 in
    let base = i * cols in
    for j = 0 to cols - 1 do
      acc := !acc +. w.(base + j) *. x.(j)
    done;
    !acc)

(** Transform bank state through learned W *)
let transform model state =
  mat_vec model.w ~rows:model.state_dim ~cols:model.state_dim state

(** State → logits: dot product of transformed state with drive signatures *)
let listen model transformed =
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to model.n_osc - 1 do
      acc := !acc +. transformed.(i) *. sig_.(i)
               +. transformed.(model.n_osc + i) *. sig_.(i)
    done;
    !acc
  ) model.drive

let softmax v =
  let mx = Array.fold_left Float.max neg_infinity v in
  let exps = Array.map (fun l -> exp (l -. mx)) v in
  let s = Array.fold_left ( +. ) 0.0 exps in
  Array.map (fun e -> e /. s) exps

let cross_entropy ~target probs =
  -. log (Float.max 1e-10 probs.(target))

(** Gradient of loss w.r.t. W.
    d_loss/d_W = outer(d_loss/d_transformed, state)
    d_loss/d_transformed = drive^T × (prob - one_hot)

    Then W -= lr × gradient. Local, no global backward pass. *)
let update_w model ~state ~transformed ~probs ~target ~lr =
  let dim = model.state_dim in
  (* d_loss/d_logits = prob - one_hot *)
  let d_logits = Array.mapi (fun i p ->
    p -. (if i = target then 1.0 else 0.0)) probs in
  (* d_loss/d_transformed = drive^T × d_logits *)
  let d_transformed = Array.init dim (fun j ->
    let acc = ref 0.0 in
    Array.iteri (fun b sig_ ->
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-8 then begin
        let s = sig_.(j mod model.n_osc) in
        acc := !acc +. dl *. s
      end
    ) model.drive;
    !acc
  ) in
  (* W -= lr × outer(d_transformed, state) *)
  for i = 0 to dim - 1 do
    let di = d_transformed.(i) in
    if Float.abs di > 1e-8 then begin
      let base = i * dim in
      for j = 0 to dim - 1 do
        model.w.(base + j) <- model.w.(base + j) -. lr *. di *. state.(j)
      done
    end
  done;
  (* Also update drive from the logit gradient *)
  let _ = transformed in
  Array.iteri (fun b sig_ ->
    let dl = d_logits.(b) in
    if Float.abs dl > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. lr *. dl *.
          (transformed.(j) +. transformed.(model.n_osc + j))
      ) sig_
  ) model.drive

(** Train on a sequence. Returns average loss. *)
let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let scaled_lr = lr /. Float.of_int seq_len in
  let drives = Array.map (fun tok -> model.drive.(tok)) tokens in
  let states = Bank.encode model.oscillators model.kernels drives in
  let total_loss = ref 0.0 in
  for t = 0 to seq_len - 2 do
    let state = states.(t) in
    let transformed = transform model state in
    let logits = listen model transformed in
    let probs = softmax logits in
    total_loss := !total_loss +. cross_entropy ~target:tokens.(t + 1) probs;
    update_w model ~state ~transformed ~probs ~target:tokens.(t + 1) ~lr:scaled_lr
  done;
  !total_loss /. Float.of_int (seq_len - 1)

(** Generate *)
let generate model seed ~n_gen ~temperature =
  let context = Array.copy seed in
  let buf = Buffer.create n_gen in
  for _ = 1 to n_gen do
    let drives = Array.map (fun tok -> model.drive.(tok)) context in
    let states = Bank.encode model.oscillators model.kernels drives in
    let state = states.(Array.length context - 1) in
    let transformed = transform model state in
    let logits = listen model transformed in
    let scaled = Array.map (fun l -> l /. temperature) logits in
    let probs = softmax scaled in
    let r = Random.float 1.0 in
    let next = ref 0 in
    let acc = ref 0.0 in
    while !acc < r && !next < vocab_size - 1 do
      acc := !acc +. probs.(!next);
      if !acc < r then incr next
    done;
    Buffer.add_char buf
      (if !next >= 32 && !next < 127 then Char.chr !next else '.');
    let n = Array.length context in
    Array.blit context 1 context 0 (n - 1);
    context.(n - 1) <- !next
  done;
  Buffer.contents buf

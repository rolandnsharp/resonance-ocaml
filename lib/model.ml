(** Text model — FFT bank → learned W → weight-tied readout.

    Optimized: accumulate gradients over the sequence,
    update weights once at the end. Not 127 times. *)

let vocab_size = 256

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
  let oscillators = Oscillator.spread n_osc in
  {
    oscillators;
    drive = Array.init vocab_size (fun _ ->
      Array.init n_osc (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    w = Array.init (state_dim * state_dim) (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale);
    kernels = Bank.precompute_kernels oscillators seq_len;
    n_osc; state_dim; seq_len;
  }

let transform model state =
  let dim = model.state_dim in
  Array.init dim (fun i ->
    let acc = ref 0.0 in
    let base = i * dim in
    for j = 0 to dim - 1 do
      acc := !acc +. model.w.(base + j) *. state.(j)
    done;
    !acc)

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

(** Train: accumulate gradients over sequence, update once. *)
let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in
  let dim = model.state_dim in
  let n_osc = model.n_osc in

  let drives = Array.map (fun tok -> model.drive.(tok)) tokens in
  let states = Bank.encode model.oscillators model.kernels drives in

  (* Accumulate W gradient and drive gradient *)
  let w_grad = Array.make (dim * dim) 0.0 in
  let drive_grad = Array.init vocab_size (fun _ -> Array.make n_osc 0.0) in
  let total_loss = ref 0.0 in

  for t = 0 to seq_len - 2 do
    let state = states.(t) in
    let target = tokens.(t + 1) in
    let transformed = transform model state in
    let logits = listen model transformed in
    let probs = softmax logits in
    total_loss := !total_loss +. cross_entropy ~target probs;

    (* d_loss/d_logits *)
    let d_logits = Array.mapi (fun i p ->
      p -. (if i = target then 1.0 else 0.0)) probs in

    (* d_loss/d_transformed = drive^T × d_logits *)
    let d_trans = Array.init dim (fun j ->
      let acc = ref 0.0 in
      for b = 0 to vocab_size - 1 do
        let dl = d_logits.(b) in
        if Float.abs dl > 1e-8 then
          acc := !acc +. dl *. model.drive.(b).(j mod n_osc)
      done;
      !acc
    ) in

    (* Accumulate W gradient: outer(d_trans, state) *)
    for i = 0 to dim - 1 do
      let di = d_trans.(i) in
      if Float.abs di > 1e-8 then begin
        let base = i * dim in
        for j = 0 to dim - 1 do
          w_grad.(base + j) <- w_grad.(base + j) +. di *. state.(j)
        done
      end
    done;

    (* Accumulate drive gradient from output error *)
    for b = 0 to vocab_size - 1 do
      let dl = d_logits.(b) in
      if Float.abs dl > 1e-6 then begin
        let dg = drive_grad.(b) in
        for j = 0 to n_osc - 1 do
          dg.(j) <- dg.(j) +. dl *. (transformed.(j) +. transformed.(n_osc + j))
        done
      end
    done
  done;

  (* Single weight update *)
  let scale = lr /. Float.of_int seq_len in
  Array.iteri (fun i g -> model.w.(i) <- model.w.(i) -. scale *. g) w_grad;
  Array.iteri (fun b dg ->
    let sig_ = model.drive.(b) in
    Array.iteri (fun j g -> sig_.(j) <- sig_.(j) -. scale *. g) dg
  ) drive_grad;

  !total_loss /. Float.of_int (seq_len - 1)

let generate model seed ~n_gen ~temperature =
  let buf = Buffer.create n_gen in
  let context = Array.copy seed in
  for _ = 1 to n_gen do
    let drives = Array.map (fun tok -> model.drive.(tok)) context in
    let states = Bank.encode model.oscillators model.kernels drives in
    let state = states.(Array.length context - 1) in
    let logits = listen model (transform model state) in
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

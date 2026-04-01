(** Text model — the whole thing, clean.

    byte_sequence
    |> lookup drive signatures
    |> FFT convolve through oscillator bank
    |> dot product with drive signatures (weight-tied)
    |> softmax → next byte prediction

    No PC layers. No coupling. No settling.
    Just: oscillator bank + weight-tied readout.
    This is what actually works. Everything else was noise. *)

let vocab_size = 256

type t = {
  oscillators : Oscillator.t array;
  drive : float array array;  (** 256 × n_osc: byte → resonance signature *)
  n_osc : int;
  state_dim : int;
}

let create n_osc =
  let state_dim = 2 * n_osc in
  let scale = 1.0 /. sqrt (Float.of_int n_osc) in
  {
    oscillators = Oscillator.spread n_osc;
    drive = Array.init vocab_size (fun _ ->
      Array.init n_osc (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    n_osc;
    state_dim;
  }

(** Byte → drive forces (lookup) *)
let strike model token = model.drive.(token)

(** State → logits: dot product with every byte's drive signature.
    Only uses position components for the dot product. *)
let listen model state =
  Array.map (fun sig_ ->
    let acc = ref 0.0 in
    for i = 0 to model.n_osc - 1 do
      acc := !acc +. state.(i) *. sig_.(i)
               +. state.(model.n_osc + i) *. sig_.(i)
    done;
    !acc
  ) model.drive

(** Softmax *)
let softmax logits =
  let mx = Array.fold_left Float.max neg_infinity logits in
  let exps = Array.map (fun l -> exp (l -. mx)) logits in
  let s = Array.fold_left ( +. ) 0.0 exps in
  Array.map (fun e -> e /. s) exps

(** Cross-entropy loss *)
let cross_entropy ~target probs =
  -. log (Float.max 1e-10 probs.(target))

(** Output error: prob - one_hot *)
let output_error ~target probs =
  Array.mapi (fun i p -> p -. (if i = target then 1.0 else 0.0)) probs

(** Update drive weights from output error.
    Both directions: as output (all bytes) and as input (current byte). *)
let update_drives model ~token ~target:_ ~output_err ~state ~lr =
  let top = state in
  (* Output direction: all drive signatures adjust *)
  Array.iteri (fun b sig_ ->
    let eb = output_err.(b) in
    if Float.abs eb > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. lr *. eb *. (top.(j) +. top.(model.n_osc + j))
      ) sig_
  ) model.drive;
  (* Input direction: current token's drive adjusts from state error *)
  let dr = model.drive.(token) in
  Array.iteri (fun j _ ->
    dr.(j) <- dr.(j) +. lr *. 0.01 *. top.(j)
  ) dr

(** Process a full sequence. Returns average loss.

    1. Map tokens to drives
    2. FFT convolve through bank → states at every position
    3. At each position: listen, compute loss, update drives *)
let train_sequence model tokens ~lr =
  let seq_len = Array.length tokens in

  (* Step 1: token → drives *)
  let drives = Array.map (strike model) tokens in

  (* Step 2: FFT convolve → full causal states *)
  let states = Bank.encode model.oscillators drives in

  (* Step 3: predict and learn at each position *)
  let total_loss = ref 0.0 in
  for t = 0 to seq_len - 2 do
    let state = states.(t) in
    let target = tokens.(t + 1) in
    let logits = listen model state in
    let probs = softmax logits in
    let loss = cross_entropy ~target probs in
    total_loss := !total_loss +. loss;
    let err = output_error ~target probs in
    update_drives model ~token:tokens.(t) ~target ~output_err:err ~state ~lr
  done;
  !total_loss /. Float.of_int (seq_len - 1)

(** Generate: given a seed sequence, predict next bytes *)
let generate model seed ~n_gen ~temperature =
  let context = Array.copy seed in
  let generated = Buffer.create n_gen in
  for _ = 1 to n_gen do
    let drives = Array.map (strike model) context in
    let states = Bank.encode model.oscillators drives in
    let state = states.(Array.length context - 1) in
    let logits = listen model state in
    let scaled = Array.map (fun l -> l /. temperature) logits in
    let probs = softmax scaled in
    (* Sample *)
    let r = Random.float 1.0 in
    let next = ref 0 in
    let acc = ref 0.0 in
    while !acc < r && !next < vocab_size - 1 do
      acc := !acc +. probs.(!next);
      if !acc < r then incr next
    done;
    Buffer.add_char generated
      (if !next >= 32 && !next < 127 then Char.chr !next else '.');
    (* Shift context *)
    let new_ctx = Array.init (Array.length context) (fun i ->
      if i < Array.length context - 1 then context.(i + 1) else !next
    ) in
    Array.blit new_ctx 0 context 0 (Array.length context)
  done;
  Buffer.contents generated

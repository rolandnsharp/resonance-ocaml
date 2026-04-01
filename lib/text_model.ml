(** Text model — strike and listen are the same resonance.

    Input:  drive[token] strikes the oscillator bank.
    Output: which drive[byte] best matches the top layer's ringing?

    The drive weights serve both directions — a bell that rings at
    440 Hz when struck is the bell you listen for when you hear 440 Hz.
    No separate output matrix. The embedding IS the readout.

    The model never resets. Oscillators ring continuously.
    The PC layers accumulate context like the brain accumulates memory.
    Decay is the only forgetting. *)

let vocab_size = 256

type t = {
  bank : Bank.t;
  drive : float array array;   (** byte → resonance signature (n_osc) *)
  mutable pc : Predictive.t;
  mutable state : Vec.t;       (** oscillator bank state *)
  n_osc : int;
}

let create ~n_osc ~n_layers =
  let state_dim = 2 * n_osc in
  let top_dim = n_osc in
  let scale = 1.0 /. sqrt (Float.of_int n_osc) in
  let dims = Array.init (n_layers + 1) (fun i ->
    if i < n_layers then state_dim else top_dim
  ) in
  {
    bank = Bank.create n_osc;
    drive = Array.init vocab_size (fun _ ->
      Array.init n_osc (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    pc = Predictive.create dims;
    state = Vec.zeros state_dim;
    n_osc;
  }

(** Logits: how well does the top layer's ringing match each byte's signature?
    logit_b = dot(top_activity, drive[b]) *)
let logits model =
  let top = Predictive.top_activity model.pc in
  Array.map (fun signature -> Vec.dot top signature) model.drive

(** Train on one (token, target) pair. Returns loss.

    1. Strike: token drives the bank
    2. Settle: PC layers find equilibrium
    3. Listen: compare ringing to all byte signatures
    4. Learn: retune drive weights from both directions *)
let train_token model ~token ~target ~settle_iters ~activity_lr ~weight_lr =
  (* 1. Strike *)
  let force = model.drive.(token) in
  model.state <- Bank.strike model.bank model.state force;

  (* 2. Settle *)
  let st = model.state in
  let pc = ref model.pc in
  for _ = 1 to settle_iters do
    pc := Predictive.step !pc st ~activity_lr ~weight_lr:0.0
  done;
  model.pc <- !pc;

  (* 3. Listen: match ringing against all byte signatures *)
  let raw_logits = logits model in
  let probs = Vec.softmax raw_logits in
  let loss = Vec.cross_entropy ~target probs in

  (* 4. Learn *)
  (* Output error: prob - one_hot *)
  let output_error = Vec.mapi (fun i p ->
    p -. (if i = target then 1.0 else 0.0)) probs in

  (* Error flowing to top PC layer: drive^T × output_error *)
  let top = Predictive.top_activity model.pc in
  let top_error = Vec.zeros model.n_osc in
  Array.iteri (fun b signature ->
    let err_b = output_error.(b) in
    if Float.abs err_b > 1e-8 then
      Array.iteri (fun j s_j ->
        top_error.(j) <- top_error.(j) +. err_b *. s_j
      ) signature
  ) model.drive;

  (* Inject error into top PC layer *)
  let top_idx = Array.length model.pc.layers - 1 in
  let layers = Array.copy model.pc.layers in
  layers.(top_idx) <- {
    Predictive.error = top_error;
    activity = Vec.add top (Vec.scale (-.weight_lr *. 10.0) top_error);
  };
  model.pc <- { model.pc with layers };

  (* Settle with weight updates *)
  for _ = 1 to 3 do
    model.pc <- Predictive.step model.pc st ~activity_lr ~weight_lr
  done;

  (* Update drive weights from BOTH directions:
     Output: drive[b] should produce correct logit for byte b
     Input: drive[token] should strike the bank correctly *)

  (* Output direction: all bytes' signatures adjust *)
  Array.iteri (fun b signature ->
    let err_b = output_error.(b) in
    if Float.abs err_b > 1e-6 then
      Array.iteri (fun j _ ->
        signature.(j) <- signature.(j) -. weight_lr *. err_b *. top.(j)
      ) signature
  ) model.drive;

  (* Input direction: current token's signature adjusts from PC bottom error *)
  let bottom_err = Predictive.bottom_error model.pc in
  let drive_row = model.drive.(token) in
  Array.iteri (fun i v ->
    drive_row.(i) <- v -. weight_lr *. bottom_err.(i)
  ) drive_row;

  loss

(** Inference: strike, settle, listen *)
let infer model token =
  let force = model.drive.(token) in
  model.state <- Bank.strike model.bank model.state force;
  let st = model.state in
  let pc = ref model.pc in
  for _ = 1 to 3 do
    pc := Predictive.step !pc st ~activity_lr:0.1 ~weight_lr:0.0
  done;
  model.pc <- !pc;
  logits model

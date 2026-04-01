(** Text model — strike, inhibit, resonate, predict.

    All layers are [pos, vel] pairs. Drive weights are [pos, vel] pairs.
    Lateral inhibition sharpens every representation.
    Strike and listen use the same resonance signature. *)

let vocab_size = 256

type t = {
  bank : Bank.t;
  drive : float array array;   (** byte → resonance signature (2*n_osc) *)
  mutable pc : Predictive.t;
  mutable state : Vec.t;
  n_osc : int;
  state_dim : int;
}

let create ~n_osc ~n_layers =
  let state_dim = 2 * n_osc in
  let scale = 1.0 /. sqrt (Float.of_int state_dim) in
  (* All layers are state_dim = 2*n_osc *)
  let dims = Array.make (n_layers + 1) state_dim in
  {
    bank = Bank.create n_osc;
    drive = Array.init vocab_size (fun _ ->
      Array.init state_dim (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    pc = Predictive.create dims;
    state = Vec.zeros state_dim;
    n_osc;
    state_dim;
  }

(** Logits via weight-tied drive: logit_b = dot(top, drive[b]) *)
let logits model =
  let top = Predictive.top_activity model.pc in
  Array.map (fun sig_ -> Vec.dot top sig_) model.drive

(** Train one token *)
let train_token model ~token ~target ~settle_iters ~activity_lr ~weight_lr =
  (* Strike the bank *)
  let force = Array.init model.n_osc (fun i -> model.drive.(token).(i)) in
  model.state <- Bank.strike model.bank model.state force;

  (* Sharpen the bank state via lateral inhibition *)
  model.state <- Inhibit.sharpen ~n_osc:model.n_osc model.state;

  (* Settle PC layers with inhibition (iPC) *)
  let st = model.state in
  for _ = 1 to settle_iters do
    model.pc <- Predictive.step model.pc st ~activity_lr ~weight_lr:0.0
  done;

  (* Logits and loss *)
  let raw_logits = logits model in
  let probs = Vec.softmax raw_logits in
  let loss = Vec.cross_entropy ~target probs in

  (* Output error *)
  let output_error = Vec.mapi (fun i p ->
    p -. (if i = target then 1.0 else 0.0)) probs in

  (* Error → top layer, settle with learning *)
  let top = Predictive.top_activity model.pc in
  let top_error = Vec.zeros model.state_dim in
  Array.iteri (fun b sig_ ->
    let eb = output_error.(b) in
    if Float.abs eb > 1e-8 then
      Array.iteri (fun j s -> top_error.(j) <- top_error.(j) +. eb *. s) sig_
  ) model.drive;

  let top_idx = model.pc.n_layers - 1 in
  let layers = Array.copy model.pc.layers in
  layers.(top_idx) <- {
    layers.(top_idx) with
    Predictive.error = top_error;
    activity = Vec.add top (Vec.scale (-.weight_lr *. 10.0) top_error);
  };
  model.pc <- { model.pc with layers };

  (* Settle with weight updates *)
  for _ = 1 to 3 do
    model.pc <- Predictive.step model.pc st ~activity_lr ~weight_lr
  done;

  (* Update drive weights — both directions *)
  Array.iteri (fun b sig_ ->
    let eb = output_error.(b) in
    if Float.abs eb > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. weight_lr *. eb *. top.(j)
      ) sig_
  ) model.drive;

  let bottom_err = Predictive.bottom_error model.pc in
  let dr = model.drive.(token) in
  Array.iteri (fun i v -> dr.(i) <- v -. weight_lr *. bottom_err.(i)) dr;

  loss

(** Inference *)
let infer model token =
  let force = Array.init model.n_osc (fun i -> model.drive.(token).(i)) in
  model.state <- Bank.strike model.bank model.state force;
  model.state <- Inhibit.sharpen ~n_osc:model.n_osc model.state;
  let st = model.state in
  for _ = 1 to 3 do
    model.pc <- Predictive.step model.pc st ~activity_lr:0.01 ~weight_lr:0.0
  done;
  logits model

(** Text model — the full pipeline in named steps.

    strike → sharpen → settle → listen → learn

    Strike and listen use the same drive signature.
    Each step is a named function you can read and reason about. *)

let vocab_size = 256

type t = {
  bank : Bank.t;
  drive : float array array;
  mutable pc : Predictive.t;
  mutable state : Vec.t;
  n_osc : int;
  state_dim : int;
}

let create ~n_osc ~n_layers =
  let state_dim = 2 * n_osc in
  let scale = 1.0 /. sqrt (Float.of_int state_dim) in
  let dims = Array.make (n_layers + 1) state_dim in
  {
    bank = Bank.create n_osc;
    drive = Array.init vocab_size (fun _ ->
      Array.init state_dim (fun _ -> (Random.float 2.0 -. 1.0) *. scale));
    pc = Predictive.create dims;
    state = Vec.zeros state_dim;
    n_osc; state_dim;
  }

(** Extract forces from a byte's drive signature (first n_osc components) *)
let drive_forces m token =
  Array.init m.n_osc (fun i -> m.drive.(token).(i))

(** Strike the bank with a byte's force pattern *)
let strike m token =
  m.state <- Bank.strike m.bank m.state (drive_forces m token);
  m.state <- Inhibit.sharpen ~n_osc:m.n_osc m.state

(** Settle the PC layers toward equilibrium *)
let settle m ~iters ~activity_lr =
  let input = m.state in
  let pc = ref m.pc in
  for _ = 1 to iters do
    pc := Predictive.step ~activity_lr ~weight_lr:0.0 input !pc
  done;
  m.pc <- !pc

(** Listen: compare top layer ringing against every byte's signature *)
let listen m =
  let top = Predictive.top_activity m.pc in
  Array.map (fun sig_ -> Vec.dot top sig_) m.drive

(** Compute output error: softmax probability minus one-hot target *)
let output_error ~target logits =
  let probs = Vec.softmax logits in
  let err = Vec.mapi (fun i p ->
    p -. (if i = target then 1.0 else 0.0)) probs in
  (probs, err)

(** Project output error back to top PC layer through drive weights.
    top_error = drive^T × output_error *)
let project_error_to_top m output_err =
  let top_err = Vec.zeros m.state_dim in
  Array.iteri (fun b sig_ ->
    let eb = output_err.(b) in
    if Float.abs eb > 1e-8 then
      Array.iteri (fun j s ->
        top_err.(j) <- top_err.(j) +. eb *. s) sig_
  ) m.drive;
  top_err

(** Inject error into top PC layer and nudge activity *)
let inject_top_error m top_err ~lr =
  let top_idx = m.pc.n_layers - 1 in
  let layers = Array.copy m.pc.layers in
  let top_act = Predictive.top_activity m.pc in
  layers.(top_idx) <- { layers.(top_idx) with
    Predictive.error = top_err;
    activity = Vec.add top_act (Vec.scale (-.lr *. 10.0) top_err);
  };
  m.pc <- { m.pc with layers }

(** Settle PC layers with weight updates (learning phase).
    Preserves injected top error by adding it back after each step. *)
let settle_and_learn m ~iters ~activity_lr ~weight_lr =
  let input = m.state in
  let top_idx = m.pc.n_layers - 1 in
  let saved_top_err = m.pc.layers.(top_idx).Predictive.error in
  for _ = 1 to iters do
    m.pc <- Predictive.step ~activity_lr ~weight_lr input m.pc;
    (* Add back the output error — don't let PC overwrite it *)
    let layers = Array.copy m.pc.layers in
    layers.(top_idx) <- { layers.(top_idx) with
      Predictive.error = Vec.add m.pc.layers.(top_idx).error saved_top_err };
    m.pc <- { m.pc with layers }
  done

(** Update drive signatures from output error.
    Each byte's signature adjusts to produce better logits. *)
let update_drives_from_output m output_err =
  let top = Predictive.top_activity m.pc in
  Array.iteri (fun b sig_ ->
    let eb = output_err.(b) in
    if Float.abs eb > 1e-6 then
      Array.iteri (fun j _ ->
        sig_.(j) <- sig_.(j) -. 0.001 *. eb *. top.(j)) sig_
  ) m.drive

(** Update current token's drive from bottom PC error.
    The bank's error tells us how to re-strike next time. *)
let update_drive_from_bottom m token ~lr =
  let err = Predictive.bottom_error m.pc in
  let dr = m.drive.(token) in
  Array.iteri (fun i v -> dr.(i) <- v -. lr *. err.(i)) dr

(** Train on one (token, target) pair. Returns loss.

    The full pipeline:
    1. strike   — byte hits the oscillator bank
    2. settle   — PC layers find equilibrium
    3. listen   — compare ringing to all byte signatures
    4. learn    — error flows back, frequencies retune *)
let train_token m ~token ~target ~settle_iters ~activity_lr ~weight_lr =
  strike m token;
  settle m ~iters:settle_iters ~activity_lr;

  let logits = listen m in
  let probs, err = output_error ~target logits in
  let loss = Vec.cross_entropy ~target probs in

  let top_err = project_error_to_top m err in
  inject_top_error m top_err ~lr:weight_lr;
  settle_and_learn m ~iters:3 ~activity_lr ~weight_lr;
  update_drives_from_output m err;
  update_drive_from_bottom m token ~lr:weight_lr;

  loss

(** Inference: strike → settle → listen *)
let infer m token =
  strike m token;
  settle m ~iters:3 ~activity_lr:0.01;
  listen m

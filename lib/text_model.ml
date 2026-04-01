(** Text model — bytes strike oscillators, predictive coding learns.

    token |> drive |> strike bank |> settle PC layers |> predict next token

    The whole model is a pipeline of pure functions.
    Learning is local: each layer updates from its own prediction error. *)

let vocab_size = 256

type t = {
  bank : Bank.t;
  drive : Vec.mat;           (* vocab_size × n_osc: how each byte strikes *)
  mutable pc : Predictive.t; (* predictive coding layers *)
  output : Vec.mat;          (* top_dim × vocab_size: state → logits *)
  mutable state : Vec.t;     (* oscillator bank state [pos; vel] *)
  n_osc : int;
}

let create ~n_osc ~n_layers =
  let state_dim = 2 * n_osc in
  let top_dim = n_osc in
  let scale = 1.0 /. sqrt (Float.of_int n_osc) in
  let dims = Array.init (n_layers + 1) (fun i ->
    if i = 0 then state_dim
    else if i = n_layers then top_dim
    else state_dim
  ) in
  {
    bank = Bank.create n_osc;
    drive = Vec.mat_rand ~rows:vocab_size ~cols:n_osc ~scale;
    pc = Predictive.create dims;
    output = Vec.mat_rand ~rows:top_dim ~cols:vocab_size ~scale;
    state = Vec.zeros state_dim;
    n_osc;
  }

let reset model =
  model.state <- Vec.zeros (2 * model.n_osc);
  model.pc <- Predictive.create
    (Array.map (fun l -> Vec.dim l.Predictive.activity) model.pc.layers)

(** Token → force vector (select row from drive matrix) *)
let token_to_force model token =
  model.drive.(token)

(** Full forward: token strikes bank, PC settles, logits emerge *)
let forward model token ~settle_steps =
  (* Strike the bank *)
  let force = token_to_force model token in
  model.state <- Bank.strike model.bank ~decay:0.95 ~dt:0.01 model.state force;

  (* Settle predictive coding with bank state as input *)
  let pc = ref model.pc in
  let bank_state = model.state in
  for _ = 1 to settle_steps do
    let net = !pc in
    pc := Predictive.step ~activity_lr:0.1 ~weight_lr:0.0 net bank_state
  done;
  model.pc <- !pc;

  (* Project top layer to logits *)
  Vec.mat_t_vec model.output (Predictive.top_activity model.pc)

(** Learn from one token prediction *)
let learn model ~target ~logits ~lr =
  (* Output error: prob - one_hot *)
  let probs = Vec.softmax logits in
  let output_error = Vec.mapi (fun i p ->
    p -. (if i = target then 1.0 else 0.0)
  ) probs in

  (* Update output weights *)
  let top_act = Predictive.top_activity model.pc in
  Vec.mat_outer_update ~lr:(-.lr) model.output top_act output_error;

  (* Inject error into top PC layer and settle with learning *)
  let top_idx = Array.length model.pc.layers - 1 in
  let layers = Array.copy model.pc.layers in
  let top_feedback = Vec.mat_vec model.output output_error in
  layers.(top_idx) <- {
    error = top_feedback;
    activity = Vec.add layers.(top_idx).activity
      (Vec.scale (-.lr *. 10.0) top_feedback);
  };
  model.pc <- { model.pc with layers };

  (* Let PC network settle with weight updates *)
  let st = model.state in
  for _ = 1 to 3 do
    model.pc <- Predictive.step ~activity_lr:0.1 ~weight_lr:lr model.pc st
  done;

  (* Update drive weights: error at bottom tells us how to re-strike *)
  let bottom_err = Predictive.bottom_error model.pc in
  let drive_row = model.drive.(target) in
  Vec.mapi (fun i d ->
    d -. lr *. bottom_err.(i)
  ) drive_row
  |> Array.iteri (fun i v -> drive_row.(i) <- v)

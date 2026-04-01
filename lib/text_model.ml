(** Text model — bytes strike oscillators, predictive coding learns.

    token |> drive |> strike bank |> settle PC layers |> predict next token *)

let vocab_size = 256

type t = {
  bank : Bank.t;
  drive : float array array;    (* vocab_size × n_osc: byte → force *)
  mutable pc : Predictive.t;
  output : Vec.mat;              (* top_dim × vocab_size *)
  mutable state : Vec.t;         (* oscillator [pos; vel] *)
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
  let drive = Array.init vocab_size (fun _ ->
    Array.init n_osc (fun _ -> (Random.float 2.0 -. 1.0) *. scale)
  ) in
  {
    bank = Bank.create n_osc;
    drive;
    pc = Predictive.create dims;
    output = Vec.mat_rand ~rows:top_dim ~cols:vocab_size ~scale;
    state = Vec.zeros state_dim;
    n_osc;
  }

let reset model =
  model.state <- Vec.zeros (2 * model.n_osc);
  model.pc <- Predictive.create
    (Array.map (fun l -> Vec.dim l.Predictive.activity) model.pc.layers)

let forward model token ~settle_steps =
  let force = model.drive.(token) in
  model.state <- Bank.strike model.bank ~decay:0.95 ~dt:0.01 model.state force;
  let st = model.state in
  let pc = ref model.pc in
  for _ = 1 to settle_steps do
    pc := Predictive.step !pc st ~activity_lr:0.1 ~weight_lr:0.0
  done;
  model.pc <- !pc;
  Vec.mat_t_vec model.output (Predictive.top_activity model.pc)

let learn model ~target ~logits ~lr =
  let probs = Vec.softmax logits in
  let output_error = Vec.mapi (fun i p ->
    p -. (if i = target then 1.0 else 0.0)
  ) probs in

  let top_act = Predictive.top_activity model.pc in
  Vec.mat_outer_update ~lr:(-.lr) model.output top_act output_error;

  let top_idx = Array.length model.pc.layers - 1 in
  let layers = Array.copy model.pc.layers in
  let top_feedback = Vec.mat_vec model.output output_error in
  layers.(top_idx) <- {
    error = top_feedback;
    activity = Vec.add layers.(top_idx).activity
      (Vec.scale (-.lr *. 10.0) top_feedback);
  };
  model.pc <- { model.pc with layers };

  let st = model.state in
  for _ = 1 to 3 do
    model.pc <- Predictive.step model.pc st ~activity_lr:0.1 ~weight_lr:lr
  done;

  let bottom_err = Predictive.bottom_error model.pc in
  let drive_row = model.drive.(target) in
  for i = 0 to model.n_osc - 1 do
    drive_row.(i) <- drive_row.(i) -. lr *. bottom_err.(i)
  done

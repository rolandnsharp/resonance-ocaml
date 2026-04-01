(** Predictive coding — learning through local prediction errors.

    Each layer predicts the layer below.
    Error = actual - predicted.
    Weights update: Δw = lr × error ⊗ activity (Hebbian).

    No backprop. No global loss signal. Each layer is autonomous. *)

type layer = {
  activity : Vec.t;
  error : Vec.t;
}

type t = {
  layers : layer array;
  weights : Vec.mat array;
}

let create dims =
  let n = Array.length dims in
  let scale = 0.1 in
  {
    layers = Array.init n (fun i ->
      { activity = Vec.zeros dims.(i);
        error = Vec.zeros dims.(i) });
    weights = Array.init (n - 1) (fun i ->
      Vec.mat_rand ~rows:dims.(i) ~cols:dims.(i + 1) ~scale);
  }

let predict t i =
  Vec.mat_t_vec t.weights.(i) t.layers.(i).activity

let compute_errors t =
  let n = Array.length t.layers in
  let layers = Array.copy t.layers in
  for i = 1 to n - 1 do
    let predicted = predict t (i - 1) in
    layers.(i) <- { layers.(i) with error = Vec.sub layers.(i).activity predicted }
  done;
  { t with layers }

let settle ~lr t =
  let n = Array.length t.layers in
  let layers = Array.copy t.layers in
  for i = 1 to n - 2 do
    let own_err = t.layers.(i).error in
    let feedback =
      if i < n - 1 then
        Vec.mat_vec t.weights.(i) t.layers.(i + 1).error
      else
        Vec.zeros (Vec.dim t.layers.(i).activity)
    in
    let new_act = Vec.add t.layers.(i).activity
      (Vec.scale lr (Vec.sub feedback own_err)) in
    layers.(i) <- { layers.(i) with activity = new_act }
  done;
  { t with layers }

let learn ~lr t =
  let n = Array.length t.layers in
  for i = 0 to n - 2 do
    Vec.mat_outer_update ~lr:(-.lr)
      t.weights.(i)
      t.layers.(i).activity
      t.layers.(i + 1).error
  done;
  t

let clamp state t =
  let layers = Array.copy t.layers in
  layers.(0) <- { layers.(0) with activity = state };
  { t with layers }

let step ~activity_lr ~weight_lr (net : t) (inp : Vec.t) =
  net
  |> clamp inp
  |> compute_errors
  |> settle ~lr:activity_lr
  |> compute_errors
  |> learn ~lr:weight_lr

let top_activity t =
  t.layers.(Array.length t.layers - 1).activity

let bottom_error t =
  t.layers.(0).error

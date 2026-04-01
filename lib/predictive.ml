(** Predictive coding layer.

    Each layer has:
    - Representational neurons (encode predictions for the layer below)
    - Error neurons (compute prediction - actual)
    - Weights (connect representational neurons between layers)

    Learning is LOCAL. Each weight updates from:
      Δw = lr * error * activity

    No backprop. No global loss. No backward pass.
    Each layer only sees its neighbors. *)

(** A vector of neuron activations *)
type state = float array

(** Weights connecting two layers *)
type weights = {
  w : float array array;  (** w.(i).(j) = weight from layer above neuron i to prediction of neuron j *)
  rows : int;
  cols : int;
}

let create_weights ~from_dim ~to_dim =
  (* Xavier initialization *)
  let scale = sqrt (2.0 /. Float.of_int (from_dim + to_dim)) in
  let w = Array.init from_dim (fun _ ->
    Array.init to_dim (fun _ ->
      (Random.float 2.0 -. 1.0) *. scale
    )
  ) in
  { w; rows = from_dim; cols = to_dim }

(** Predict layer below from layer above: prediction = W^T * activity *)
let predict weights (activity : state) : state =
  let pred = Array.make weights.cols 0.0 in
  for j = 0 to weights.cols - 1 do
    let sum = ref 0.0 in
    for i = 0 to weights.rows - 1 do
      sum := !sum +. weights.w.(i).(j) *. activity.(i)
    done;
    pred.(j) <- !sum
  done;
  pred

(** Compute prediction error: error = actual - predicted *)
let error (actual : state) (predicted : state) : state =
  Array.init (Array.length actual) (fun i ->
    actual.(i) -. predicted.(i)
  )

(** Propagate error upward: feedback = W * error *)
let propagate_error weights (err : state) : state =
  let feedback = Array.make weights.rows 0.0 in
  for i = 0 to weights.rows - 1 do
    let sum = ref 0.0 in
    for j = 0 to weights.cols - 1 do
      sum := !sum +. weights.w.(i).(j) *. err.(j)
    done;
    feedback.(i) <- !sum
  done;
  feedback

(** Update activity based on local prediction errors.
    dx/dt = -own_error + feedback_from_below

    This is the core of predictive coding:
    each neuron finds a compromise between matching its
    top-down prediction and predicting the layer below. *)
let update_activity ~lr ~(own_error : state) ~(feedback : state) (activity : state) : state =
  Array.init (Array.length activity) (fun i ->
    let de = if i < Array.length own_error then own_error.(i) else 0.0 in
    let fb = if i < Array.length feedback then feedback.(i) else 0.0 in
    activity.(i) +. lr *. (-. de +. fb)
  )

(** Update weights: Δw = lr * error * activity (Hebbian) *)
let update_weights ~lr weights (err : state) (activity : state) =
  for i = 0 to weights.rows - 1 do
    for j = 0 to weights.cols - 1 do
      weights.w.(i).(j) <- weights.w.(i).(j)
        +. lr *. err.(j) *. activity.(i)
    done
  done


(** A complete predictive coding network *)
type layer = {
  mutable activity : state;
  mutable error_neurons : state;
}

type network = {
  layers : layer array;
  weights : weights array;  (** weights.(i) connects layer i to layer i+1 *)
}

let create_network dims =
  let n = Array.length dims in
  let layers = Array.init n (fun i ->
    { activity = Array.make dims.(i) 0.0;
      error_neurons = Array.make dims.(i) 0.0 }
  ) in
  let weights = Array.init (n - 1) (fun i ->
    create_weights ~from_dim:dims.(i) ~to_dim:dims.(i + 1)
  ) in
  { layers; weights }

(** One step of predictive coding inference + learning.

    1. Compute predictions (top-down)
    2. Compute errors (actual - predicted)
    3. Update activities (settle toward equilibrium)
    4. Update weights (Hebbian: error * activity)

    All local. No backward pass. *)
let step ~activity_lr ~weight_lr net =
  let n = Array.length net.layers in

  (* 1. Compute predictions and errors for each layer *)
  for i = 1 to n - 1 do
    let pred = predict net.weights.(i - 1) net.layers.(i - 1).activity in
    net.layers.(i).error_neurons <- error net.layers.(i).activity pred
  done;

  (* 2. Update activities: each layer settles toward equilibrium *)
  for i = 0 to n - 2 do
    let own_err = net.layers.(i).error_neurons in
    (* Feedback from layer below *)
    let feedback =
      if i < n - 1 then
        propagate_error net.weights.(i) net.layers.(i + 1).error_neurons
      else
        Array.make (Array.length net.layers.(i).activity) 0.0
    in
    net.layers.(i).activity <-
      update_activity ~lr:activity_lr ~own_error:own_err ~feedback net.layers.(i).activity
  done;

  (* 3. Update weights: local Hebbian rule *)
  for i = 0 to n - 2 do
    update_weights ~lr:weight_lr
      net.weights.(i)
      net.layers.(i + 1).error_neurons
      net.layers.(i).activity
  done

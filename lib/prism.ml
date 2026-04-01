(** Prism: butterfly mixing layers.

    Each layer: pair neighboring elements and mix them.
    Between layers: shuffle (bit-reversal permutation).
    O(log n) layers of O(n) pair-mixing = O(n log n) total.

    This IS how FFT works internally. Same math, learnable weights.
    No matrix multiply. Each pair is 4 multiply-adds.
    Every pair is independent — parallelizes across cores. *)

(** One butterfly layer: n/2 pairs, each with a 2×2 mixing matrix *)
type layer = {
  a : float array;  (** n/2 mixing coefficients *)
  b : float array;
  c : float array;
  d : float array;
  n_pairs : int;
}

type t = {
  layers : layer array;
  shuffle : int array;
  dim : int;
  n_layers : int;
}

let bit_reverse_perm n =
  let bits = max 1 (int_of_float (ceil (log (Float.of_int n) /. log 2.0))) in
  Array.init n (fun i ->
    let rev = ref 0 in
    let v = ref i in
    for _ = 0 to bits - 1 do
      rev := !rev * 2 + !v mod 2;
      v := !v / 2
    done;
    !rev mod n)

let create dim =
  let n_layers = max 2 (int_of_float (ceil (log (Float.of_int dim) /. log 2.0))) in
  let n_pairs = dim / 2 in
  let scale = 1.0 /. sqrt (Float.of_int n_layers) in
  {
    layers = Array.init n_layers (fun _ -> {
      (* Initialize near identity: a≈1, d≈1, b≈0, c≈0 *)
      a = Array.init n_pairs (fun _ -> 1.0 +. (Random.float 2.0 -. 1.0) *. scale);
      b = Array.init n_pairs (fun _ -> (Random.float 2.0 -. 1.0) *. scale);
      c = Array.init n_pairs (fun _ -> (Random.float 2.0 -. 1.0) *. scale);
      d = Array.init n_pairs (fun _ -> 1.0 +. (Random.float 2.0 -. 1.0) *. scale);
      n_pairs;
    });
    shuffle = bit_reverse_perm dim;
    dim;
    n_layers;
  }

(** Apply one butterfly layer: mix pairs *)
let apply_layer layer x =
  let n = Array.length x in
  let out = Array.make n 0.0 in
  for p = 0 to layer.n_pairs - 1 do
    let i = p * 2 and j = p * 2 + 1 in
    if j < n then begin
      out.(i) <- layer.a.(p) *. x.(i) +. layer.b.(p) *. x.(j);
      out.(j) <- layer.c.(p) *. x.(i) +. layer.d.(p) *. x.(j)
    end else
      out.(i) <- x.(i)
  done;
  out

(** Apply shuffle *)
let apply_shuffle shuffle x =
  Array.init (Array.length x) (fun i -> x.(shuffle.(i)))

(** Forward: butterfly layers interleaved with shuffles *)
let forward prism x =
  Array.fold_left (fun state layer ->
    apply_layer layer state |> apply_shuffle prism.shuffle
  ) x prism.layers

(** Backward: compute gradients and update params, return d_input *)
let backward prism x d_output ~lr =
  (* Store intermediates for backward pass *)
  let intermediates = Array.make (prism.n_layers + 1) x in
  let state = ref x in
  for i = 0 to prism.n_layers - 1 do
    state := apply_layer prism.layers.(i) !state |> apply_shuffle prism.shuffle;
    intermediates.(i + 1) <- !state
  done;

  (* Inverse shuffle for backward *)
  let inv_shuffle = Array.make prism.dim 0 in
  Array.iteri (fun src dst -> inv_shuffle.(dst) <- src) prism.shuffle;

  let d = ref d_output in
  for i = prism.n_layers - 1 downto 0 do
    (* Unshuffle gradient *)
    d := Array.init prism.dim (fun j -> !d.(inv_shuffle.(j)));

    let pre = intermediates.(i) in
    let layer = prism.layers.(i) in
    let d_pre = Array.make prism.dim 0.0 in

    for p = 0 to layer.n_pairs - 1 do
      let ii = p * 2 and jj = p * 2 + 1 in
      if jj < prism.dim then begin
        let di = !d.(ii) and dj = !d.(jj) in
        let xi = pre.(ii) and xj = pre.(jj) in

        (* Gradient of mixing params *)
        layer.a.(p) <- layer.a.(p) -. lr *. di *. xi;
        layer.b.(p) <- layer.b.(p) -. lr *. di *. xj;
        layer.c.(p) <- layer.c.(p) -. lr *. dj *. xi;
        layer.d.(p) <- layer.d.(p) -. lr *. dj *. xj;

        (* Gradient of input *)
        d_pre.(ii) <- di *. layer.a.(p) +. dj *. layer.c.(p);
        d_pre.(jj) <- di *. layer.b.(p) +. dj *. layer.d.(p)
      end else
        d_pre.(ii) <- !d.(ii)
    done;
    d := d_pre
  done;
  !d

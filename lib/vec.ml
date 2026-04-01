(** Vectors as functions. No mutation. *)

type t = float array

let create n f = Array.init n f
let zeros n = Array.make n 0.0
let dim = Array.length

let map = Array.map
let map2 f a b = Array.init (dim a) (fun i -> f a.(i) b.(i))
let mapi = Array.mapi

let add = map2 ( +. )
let sub = map2 ( -. )
let scale s = map (( *. ) s)
let hadamard = map2 ( *. )

let dot a b =
  let n = dim a in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do acc := !acc +. a.(i) *. b.(i) done;
  !acc

let sum = Array.fold_left ( +. ) 0.0
let norm_sq v = dot v v

(** Matrix as array of row vectors *)
type mat = t array

let mat_create ~rows ~cols f =
  Array.init rows (fun i -> Array.init cols (fun j -> f i j))

let mat_rand ~rows ~cols ~scale =
  mat_create ~rows ~cols (fun _ _ -> (Random.float 2.0 -. 1.0) *. scale)

(** Matrix-vector: y = M * x *)
let mat_vec m x =
  Array.map (fun row -> dot row x) m

(** Transposed matrix-vector: y = M^T * x *)
let mat_t_vec m x =
  let cols = dim m.(0) in
  Array.init cols (fun j ->
    let acc = ref 0.0 in
    Array.iteri (fun i row -> acc := !acc +. row.(j) *. x.(i)) m;
    !acc
  )

(** Outer product update: M += lr * a (outer) b *)
let mat_outer_update ~lr m a b =
  Array.iteri (fun i row ->
    Array.iteri (fun j _ ->
      row.(j) <- row.(j) +. lr *. a.(i) *. b.(j)
    ) row
  ) m

(** Softmax *)
let softmax v =
  let mx = Array.fold_left Float.max neg_infinity v in
  let exps = map (fun x -> exp (x -. mx)) v in
  let s = sum exps in
  map (fun e -> e /. s) exps

(** Cross-entropy loss *)
let cross_entropy ~target probs =
  -. log (Float.max 1e-10 probs.(target))

(** Sample from probability distribution *)
let sample ?(temperature=1.0) v =
  let scaled = map (fun x -> x /. temperature) v in
  let probs = softmax scaled in
  let r = Random.float 1.0 in
  let rec pick i acc =
    if i >= dim probs - 1 then i
    else let acc' = acc +. probs.(i) in
      if r < acc' then i else pick (i + 1) acc'
  in pick 0 0.0

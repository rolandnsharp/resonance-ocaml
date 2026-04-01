(** Parallel map using OCaml 5 Domains.

    Split work across cores. Each core processes a chunk independently.
    No locks, no shared mutable state — pure parallel map. *)

let n_cores =
  try int_of_string (Sys.getenv "CORES")
  with _ -> Domain.recommended_domain_count ()

(** Parallel map over an array. Splits into chunks, one per core. *)
let map f arr =
  let n = Array.length arr in
  if n <= 1 || n_cores <= 1 then Array.map f arr
  else begin
    let results = Array.make n (f arr.(0)) in
    let chunk_size = (n + n_cores - 1) / n_cores in
    let domains = Array.init (n_cores - 1) (fun core ->
      let lo = (core + 1) * chunk_size in
      let hi = Int.min n (lo + chunk_size) in
      Domain.spawn (fun () ->
        for i = lo to hi - 1 do
          results.(i) <- f arr.(i)
        done)
    ) in
    (* Main domain handles first chunk *)
    let hi = Int.min n chunk_size in
    for i = 0 to hi - 1 do
      results.(i) <- f arr.(i)
    done;
    (* Wait for all domains *)
    Array.iter Domain.join domains;
    results
  end

(** Parallel init — like Array.init but across cores *)
let init n f =
  if n <= 1 || n_cores <= 1 then Array.init n f
  else begin
    let dummy = f 0 in
    let results = Array.make n dummy in
    let chunk_size = (n + n_cores - 1) / n_cores in
    let domains = Array.init (n_cores - 1) (fun core ->
      let lo = (core + 1) * chunk_size in
      let hi = Int.min n (lo + chunk_size) in
      Domain.spawn (fun () ->
        for i = lo to hi - 1 do
          results.(i) <- f i
        done)
    ) in
    let hi = Int.min n chunk_size in
    for i = 0 to hi - 1 do
      results.(i) <- f i
    done;
    Array.iter Domain.join domains;
    results
  end

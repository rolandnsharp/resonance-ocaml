(** Resonance — FFT oscillator bank, batch-parallel training.

    Each step: N_BATCH sequences processed in parallel across CPU cores.
    Gradients merged, weights updated once. *)

let read_file path =
  let ic = open_in path in
  let n = in_channel_length ic in
  let s = Bytes.create n in
  really_input ic s 0 n;
  close_in ic;
  Bytes.to_string s

let env s d = try int_of_string (Sys.getenv s) with _ -> d
let env_f s d = try float_of_string (Sys.getenv s) with _ -> d

let () =
  Random.self_init ();
  let text = read_file
    (if Array.length Sys.argv > 1 then Sys.argv.(1)
     else "data/shakespeare.txt") in
  let text_len = String.length text in

  let n_osc   = env "N_OSC" 64 in
  let n_steps = env "N_STEPS" 10000 in
  let seq_len = env "SEQ_LEN" 128 in
  let batch   = env "BATCH" (Resonance.Par.n_cores) in
  let lr      = env_f "LR" 0.01 in

  Printf.printf "Resonance — FFT oscillator bank (%d cores)\n" Resonance.Par.n_cores;
  Printf.printf "Text: %d bytes | %d osc, seq=%d, batch=%d, lr=%g\n\n%!"
    text_len n_osc seq_len batch lr;

  let model = Resonance.Model.create n_osc seq_len in
  let warmdown = n_steps / 5 in

  for step = 0 to n_steps - 1 do
    let s = if step < 100 then Float.of_int step /. 100.0
      else if step >= n_steps - warmdown then
        0.5 *. (1.0 +. cos (Float.pi *. Float.of_int (step - n_steps + warmdown)
                            /. Float.of_int warmdown))
      else 1.0 in
    let cur_lr = lr *. s in

    (* Batch of random sequences — one per core *)
    let token_seqs = Array.init batch (fun _ ->
      let start = Random.int (text_len - seq_len - 1) in
      Array.init seq_len (fun i -> Char.code text.[start + i])
    ) in

    let loss = Resonance.Model.train_batch model token_seqs ~lr:cur_lr in

    if step mod 100 = 0 then begin
      let bpc = loss /. log 2.0 in
      Printf.printf "step %5d  loss %.3f  bpc %.3f  lr %.1e" step loss bpc cur_lr;
      if step mod 500 = 0 then begin
        let seed_str = "First Citizen:\n" in
        let seed = Array.init (String.length seed_str) (fun i -> Char.code seed_str.[i]) in
        let gen = Resonance.Model.generate model seed ~n_gen:60 ~temperature:0.8 in
        Printf.printf "  | %s" gen
      end;
      Printf.printf "\n%!"
    end
  done;
  Printf.printf "\nDone.\n"

let () =
  Random.self_init ();
  let n_osc = 64 in
  let seq_len = 128 in
  let model = Resonance.Model.create n_osc seq_len in
  let tokens = Array.init seq_len (fun _ -> Random.int 256) in

  (* Time each part *)
  let time f =
    let t0 = Unix.gettimeofday () in
    let r = f () in
    let dt = Unix.gettimeofday () -. t0 in
    (r, dt)
  in

  (* 1. Drive lookup *)
  let drives, t_drive = time (fun () ->
    Array.map (fun tok -> model.drive.(tok)) tokens) in
  Printf.printf "Drive lookup:  %.1f ms\n" (t_drive *. 1000.0);

  (* 2. FFT encode *)
  let states, t_fft = time (fun () ->
    Resonance.Bank.encode model.oscillators model.kernels drives) in
  Printf.printf "FFT encode:    %.1f ms\n" (t_fft *. 1000.0);

  (* 3. W transform + listen for all positions *)
  let _, t_infer = time (fun () ->
    Array.map (fun state ->
      let transformed = Resonance.Model.transform model state in
      Resonance.Model.listen model transformed
    ) states) in
  Printf.printf "W + listen:    %.1f ms\n" (t_infer *. 1000.0);

  (* 4. Full train step *)
  let _, t_train = time (fun () ->
    Resonance.Model.train_sequence model tokens ~lr:0.01) in
  Printf.printf "Full train:    %.1f ms\n" (t_train *. 1000.0);
  Printf.printf "Steps/sec:     %.1f\n" (1000.0 /. (t_train *. 1000.0))

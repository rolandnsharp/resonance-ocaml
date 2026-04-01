(** Resonance — continuous stream, never reset.

    The oscillators ring from the first byte to the last.
    The PC layers accumulate context like memory.
    Decay is the only forgetting.
    Strike and listen are the same resonance. *)

let read_file path =
  let ic = open_in path in
  let n = in_channel_length ic in
  let s = Bytes.create n in
  really_input ic s 0 n;
  close_in ic;
  Bytes.to_string s

let env_int name default =
  int_of_string (try Sys.getenv name with _ -> string_of_int default)
let env_float name default =
  float_of_string (try Sys.getenv name with _ -> Printf.sprintf "%g" default)

let () =
  Random.self_init ();
  let text_path = if Array.length Sys.argv > 1 then Sys.argv.(1)
    else "data/shakespeare.txt" in
  let text = if Sys.file_exists text_path then read_file text_path
    else "To be, or not to be, that is the question." in

  let n_osc    = env_int   "N_OSC"    32 in
  let n_layers = env_int   "N_LAYERS" 2 in
  let n_steps  = env_int   "N_STEPS"  50000 in
  let settle   = env_int   "SETTLE"   3 in
  let lr       = env_float "LR"       0.001 in
  let text_len = String.length text in

  Printf.printf "Resonance — continuous stream, strike and listen\n";
  Printf.printf "Text: %d bytes | %d osc, %d layers, settle=%d\n"
    text_len n_osc n_layers settle;
  Printf.printf "LR: %g | No reset. Oscillators ring forever.\n\n%!" lr;

  let model = Resonance.Text_model.create ~n_osc ~n_layers in
  let warmdown_start = n_steps - n_steps / 5 in

  let pos = ref 0 in
  let loss_sum = ref 0.0 in
  let loss_count = ref 0 in

  for step = 0 to n_steps - 1 do
    let warmup_steps = Int.min 500 (n_steps / 20) in
    let lr_scale =
      if step < warmup_steps then Float.of_int step /. Float.of_int warmup_steps
      else if step >= warmdown_start then
        let p = Float.of_int (step - warmdown_start)
          /. Float.of_int (n_steps - warmdown_start) in
        0.5 *. (1.0 +. cos (Float.pi *. p))
      else 1.0 in
    let cur_lr = lr *. lr_scale in

    let token = Char.code text.[!pos] in
    let target = Char.code text.[(!pos + 1) mod text_len] in

    let loss = Resonance.Text_model.train_token model ~token ~target
      ~settle_iters:settle
      ~activity_lr:0.01
      ~weight_lr:cur_lr in

    loss_sum := !loss_sum +. loss;
    incr loss_count;
    pos := (!pos + 1) mod text_len;

    if step mod 1000 = 0 then begin
      let avg = !loss_sum /. Float.of_int !loss_count in
      let bpc = avg /. log 2.0 in
      Printf.printf "step %5d  loss %.3f  bpc %.3f  lr %.1e  pos %d" step avg bpc cur_lr !pos;
      loss_sum := 0.0; loss_count := 0;

      if step mod 5000 = 0 then begin
        Printf.printf "  | ";
        let last = ref token in
        for _ = 1 to 60 do
          let l = Resonance.Text_model.infer model !last in
          let next = Resonance.Vec.sample ~temperature:0.8 l in
          last := next;
          Printf.printf "%c"
            (if next >= 32 && next < 127 then Char.chr next else '.')
        done
      end;
      Printf.printf "\n%!"
    end
  done;
  Printf.printf "\nDone.\n"

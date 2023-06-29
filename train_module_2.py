from happytransformer import HappyTextToText

happy_tt = HappyTextToText("T5", "t5-base")

from happytransformer import TTTrainArgs

train_args = TTTrainArgs(num_train_epochs=1, max_input_length=1024, max_output_length=1024)

happy_tt.train("./train_module_2.csv", args=train_args)

happy_tt.save("./home/netojoaquim/predict/modulo2/")
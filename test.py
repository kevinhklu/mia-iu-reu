import pandas as pd 

trained_qraw = pd.read_excel("trained_q&a.xlsx", usecols=["Question"])
trained_questions = trained_qraw["Question"].tolist()
print(trained_questions)


str = """段���帮����为��功。�为� 运势财运�� �象��子观此卦象，以处境艰难自励，穷且益坚，舍身捐命，以行其夙志。����》卦：通泰。����                              ���严�以�财星受制于戌�����财����份�
����微信�lwh791水�水�泽��泽中干涸����夬�，� 运猛增。根据卦爻原理，守常勿动，待时而动，财运自来。《泽天夬》卦：王庭里正跳舞��。�人������人�����中传��令�����"""

tmp = b"\xe2\x96\x85"

file_path = "./data/02_train_data/00_txt_for_train_tokenizer/txt_for_train_tokenizer_0027.txt"

with open(file_path, "r", encoding="utf-8") as f:
    for line in f.readlines():
        if tmp in line.encode("utf-8"):
            print(line)

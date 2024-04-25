import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import pickle


model_file = "./my_sp_bbpe_tokenizer_20G_raw/my_sp_bbpe_tokenizer.model"
# model_file = "./test.model"

# tokenizer = pickle.load(open(model_file, "rb"))

sp_model = spm.SentencePieceProcessor(model_file=model_file)

text = """<s>1<|beginoftext|>"""

ids = sp_model.Encode(text, out_type=int)
print(ids)

sp_model_proto = sp_pb2_model.ModelProto()
sp_model_proto.ParseFromString(sp_model.serialized_model_proto())

print(len(sp_model_proto.pieces))

sp_model_proto.pieces.add(piece="<|endofprompt11|>", score=0, type=sp_pb2_model.ModelProto.SentencePiece.CONTROL)
print(sp_model_proto.pieces.pop(63995))

print(len(sp_model_proto.pieces))

# SPECIAL_TUPLE = ('<s>', '</s>', '<|beginoftext|>', '<|endoftext|>', '<|endofprompt|>', '<|im_start|>', '<|im_end|>', '<|UNK|>', '<|PAD|>', '<|CLS|>', '<|SEP|>', '<|MASK|>', '<|BOS|>', '<|EOS|>')
# for piece in sp_model_proto.pieces[63995:]:
#     # if piece.type == sp_pb2_model.ModelProto.SentencePiece.USER_DEFINED:
#     #     print(piece.piece)
#     print(piece.piece, piece.type)
    # 将特殊token设置为 CONTROL 类型，之前是 USER_DEFINED 类型
    # if piece.piece in SPECIAL_TUPLE:
    #     piece.type = sp_pb2_model.ModelProto.SentencePiece.CONTROL
        
# with open("new.model", "wb") as f:
#     f.write(sp_model_proto.SerializeToString())
        
# sp_model_modified = spm.SentencePieceProcessor(model_proto=sp_model_proto.SerializeToString())

# ids = sp_model_modified.Encode(text, out_type=int)
# print(ids)

# ids = [0, 1, 2, 3, 4, 5]

# print(list(map(tokenizer.IdToPiece, [i for i in range(16)])))

# print(tokenizer.vocab_size())

# print(tokenizer.GetPieceSize())

# with open("test.model", "wb") as f:
#     pickle.dump(tokenizer, f)

import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer


tokenizer_dir = "./tokenizer/hf_bbpe_tokenizer"


chosen_input_ids = [[64000, 64003, 13007, 64012, 11229,   694,   262, 61732,  9649,    13,
        64004, 64003,  4381, 64012,  1288,  2152,  8259,  8893,  2397,   265,
        10008, 63631, 37916,   570, 25000,   576, 45438, 19576,   265,  2598,
         4082,  1427,  2351,  8893,  3490,   270,  3375,  3059,   920,  5030,
        53548, 13209,   295,  7329,   265,  3359, 11008, 31684,   570, 25000,
          576,   978,  1399, 31989, 32182,   484, 45045,   265,  2445, 37583,
         5524,  2432,  2405, 50687,   459, 64004, 64003,  4381, 64012, 25000,
          270, 19576,  8893,   314,   570,    42, 19260,   276,    75, 17604,
         7763,   722,  8246,    89,  2911,   978, 37916, 11494, 17766,  1906,
         5947,  1925, 12133,   366,  1372,    21,   454, 60794,   265,   535,
         3915,   355,  1441,   962, 18768,  1218, 56623,   265,   978, 56623,
         6997, 35185, 25000,  2366, 13670,  1847,   629,   295, 35412, 13090,
        37703,  5447,  8210,   570,  6106,   576,  7421, 19946,   265,  2002,
         7421,   794,  1467,  6199,  2305,  7033,   484,  7008,  1058,   198,
        25000,  2366,  1582,  2014, 23396, 38666,   679,  7072, 12892,   270,
         5133,   265,  8597,  8259,  4029,   416,  6100,   416, 13656,   416,
        15373,   416,  8784,  4029,   484,  5030,   520,   295, 25000, 45780,
         5349,  4071,  4083,  3425,  3718,  3898,  5030,   416,  1927,  8533,
        26207,   972,   891,  4048,   440, 20169,  7421,  1058,   198, 25000,
          346, 21743, 10435,  4500,  1022, 12696,  4739,  2164,   265,  2909,
         6971, 16216,   416, 42886,   484,  2765,   295, 25000, 12149,   270,
        14700, 21169,  6925,  7092,   265, 53154,  3869,  5349,  7899, 39737,
         1927,  7326,   265, 25000, 15765,   865,  9575,  4067,  2547,   921,
        35201,  2361,  1058,   198,  4987,   265, 25000,   270, 22535,  9991,
          528,  5440, 15162, 24038,   295,  1679,  6431,  7231, 25000,  1427,
         7291,  4699,  6106,   265, 22839,  1427],
        [64000, 64003, 13007, 64012, 11229,   694,   262, 61732,  9649,    13,
         64004, 64003,  4381, 64012,   680, 20870,   794,   795,   618,  4516,
           459, 64004, 64003,  4381, 64012,  2278, 25000,   265, 62770, 36765,
           265, 20870,  1849,   795,   618,  4516,   295,  8751,   512,  8751,
           265,  6243,   512,  6243,   295,  8751,   484, 47328, 23535,  3490,
         39445,  6121,   295,  6554,   265,  8751,   891, 20870,   795,   555,
           265,  6243, 11748,   891, 31778,  3329,  6243,  6959,   295,  7329,
           265, 13575,   480, 14479,   265,  4516,   314, 31964,  6243,   265,
          4663, 37891, 25816,   265,  1849,   865,  3375,  8751,   920, 20870,
         11829,  1058,   198,  4987,   265,  4939,  6509,  5093,  1997,  8381,
          5922,   920,  9405,  2971, 15293,   891, 20870,   795,   555, 10426,
          4516,   265,  1830,  3375,  2100,  4818, 17026,   265,   728, 61243,
         37668,   920, 36451,  4582,  4186,   295, 40548,  9564]]

rejected_input_ids = [[64000, 64003, 13007, 64012, 11229,   694,   262, 61732,  9649,    13,
        64004, 64003,  4381, 64012,  1288,  2152,  8259,  8893,  2397,   265,
        10008, 63631, 37916,   570, 25000,   576, 45438, 19576,   265,  2598,
         4082,  1427,  2351,  8893,  3490,   270,  3375,  3059,   920,  5030,
        53548, 13209,   295,  7329,   265,  3359, 11008, 31684,   570, 25000,
          576,   978,  1399, 31989, 32182,   484, 45045,   265,  2445, 37583,
         5524,  2432,  2405, 50687,   459, 64004, 64003,  4381, 64012, 47108,
          198,     7, 47108,     8, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
        64006, 64006, 64006, 64006, 64006, 64006],
        [64000, 64003, 13007, 64012, 11229,   694,   262, 61732,  9649,    13,
         64004, 64003,  4381, 64012,   680, 20870,   794,   795,   618,  4516,
           459, 64004, 64003,  4381, 64012, 17626,   265, 15869,   647,  8591,
           920,  7214,  3277,   295, 55011,  2244,  1425,   336, 39958,   484,
         60356,   270,  8141,   295, 64006, 64006, 64006, 64006, 64006, 64006,
         64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
         64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006, 64006,
         64006, 64006]]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

print("chosen 0:\n", tokenizer.decode(chosen_input_ids[0], skip_special_tokens=False))
print("rejected 0:\n", tokenizer.decode(rejected_input_ids[0], skip_special_tokens=False))
print("chosen 1:\n", tokenizer.decode(chosen_input_ids[1], skip_special_tokens=False))
print("rejected 1:\n", tokenizer.decode(rejected_input_ids[1], skip_special_tokens=False))

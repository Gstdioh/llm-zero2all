class Z2allConfig:

    def __init__(
        self,
        vocab_size=32000,
        hidden_dim=4096,
        intermediate_size=11008,  # FFN中间层的大小, 不为None, 则mutiple_of参数无效
        multiple_of=256,  # make SwiGLU hidden layer size multiple of large power of 2, 256的倍数, 影响intermediate_size（见llama.c）
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,  # 用于GQA
        hidden_act="silu",  # FFN中的激活函数
        max_seq_len=2048,
        initializer_range=0.02,  # 参数初始化时的标准差
        rms_norm_eps=1e-6,  # 防止除0的小数
        # use_cache=True,  # 是否缓存kv
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,  # 预训练时的张量并行度, tensor parallelism
        tie_word_embeddings=False,  # 是否共享word embedding和word prediction的参数
        rope_theta=10000.0,
        rope_scaling=None,  # 缩放方法，用于长度外推
        attention_bias=False,  # attention中的project是否加bias
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.mutiple_of = multiple_of
        self.n_layers = n_layers
        self.n_heads = n_heads

        # 用于GQA
        if n_kv_heads is None:
            n_kv_heads = n_heads

        self.n_kv_heads = n_kv_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        # self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

    def _rope_scaling_validation(self):
        """
        验证`rope_scaling`配置
        `rope_scaling`必须是一个字典，包含两个字段，`type`和`factor`
        `type`字段必须是['linear', 'dynamic']中的一个
        `factor`字段必须是一个大于1的浮点数
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError( 
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}"
            )

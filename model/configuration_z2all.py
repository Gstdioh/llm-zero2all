from transformers import PretrainedConfig


class Z2allConfig(PretrainedConfig):

    model_type = "z2all"

    def __init__(
        self,
        vocab_size=64000,
        hidden_dim=4096,
        intermediate_size=None,  # FFN中间层的大小, 不为None, 则mutiple_of参数无效；为None这通过hidden_dim和mutiple_of计算，即8/3*hidden_dim
        multiple_of=256,  # make SwiGLU hidden layer size multiple of large power of 2, 256的倍数, 影响intermediate_size（见llama.c）
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,  # 用于GQA
        max_seq_len=2048,
        initializer_range=0.02,  # 参数初始化时的标准差
        rms_norm_eps=1e-5,  # 防止除0的小数
        pad_token_id=64006,  # pad token <|PAD|>
        tie_word_embeddings=False,  # 是否共享word embedding和word prediction的参数
        rope_theta=10000.0,
        rope_scaling=None,  # 缩放方法，用于长度外推
        rope_interleaved=False,  # rope使用交错方式计算
        attention_bias=False,  # attention中的project是否加bias
        attention_dropout=0.0,
        dropout1=0.0,
        # drop_path1=0.0,  # 通常LLM中不会用到droppath
        dropout2=0.0,
        # drop_path2=0.0,
        residual_in_fp32=True,  # 残差连接是否使用fp32
        use_flash=True,
        use_fused_rope=True,
        use_fused_cross_entropy=True,
        use_fused_dropout_add_norm=True,
        use_fused_swiglu=True,
        **kwargs,
    ):
        self.auto_map = {
            "AutoConfig": "configuration_z2all.Z2allConfig",
            "AutoModel": "modeling_z2all.Z2allModel",
            "AutoModelForCausalLM": "modeling_z2all.Z2allForCausalLM",
        }
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.multiple_of = multiple_of
        self.n_layers = n_layers
        self.n_heads = n_heads

        # 用于GQA
        if n_kv_heads is None:
            n_kv_heads = n_heads

        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_kv_heads = n_kv_heads
        # self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleaved = rope_interleaved
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        
        self.attention_dropout = attention_dropout
        self.dropout1 = dropout1
        # self.drop_path1 = drop_path1
        self.dropout2 = dropout2
        # self.drop_path2 = drop_path2
        self.residual_in_fp32 = residual_in_fp32
        
        # 是否使用融合算子
        self.use_flash = use_flash
        self.use_fused_swiglu = use_fused_swiglu
        self.use_fused_dropout_add_norm = use_fused_dropout_add_norm
        self.use_fused_rope = use_fused_rope
        self.use_fused_cross_entropy = use_fused_cross_entropy
        
        self.pad_token_id = pad_token_id
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

    def set_auto_map(self, auto_map):
        self.auto_map = auto_map
    
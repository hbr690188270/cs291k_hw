
class BairuConfig():
    def __init__(
        self,
        vocab_size=30522,
        embedding_dim = 256,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        layernorm_embedding = True,
        use_cache=True,
        classifier_dropout=None,
        decoder_embedding_dim = 256,
        decoder_hidden_size = 256,
        decoder_hidden_layer = 2,
        batch_first = False,
        decoder_residual = True,
        decoder_init = "enc",  ## enc or none
        decoder_pe = 'sin',
        **kwargs
    ):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.layernorm_embedding = layernorm_embedding
        self.pad_token_id = pad_token_id
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.batch_first = batch_first
        self.decoder_hidden_layer = decoder_hidden_layer
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_embedding_dim = decoder_embedding_dim
        self.decoder_residual = decoder_residual
        self.decoder_init = decoder_init
        self.decoder_pe = decoder_pe



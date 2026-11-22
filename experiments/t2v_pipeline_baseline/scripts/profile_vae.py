"""Instrument the existing Wan decoder and its temporal slices."""


class VAEProfile:
    def __init__(self, vae, event):
        self.vae = vae
        self.event = event
        self.core = []
        self.chunks = []

    def install(self):
        self.original = self.vae.model.decode
        self.vae.model.decode = self.decode
        self.pre = self.vae.model.decoder.register_forward_pre_hook(self.before)
        self.post = self.vae.model.decoder.register_forward_hook(self.after)

    def restore(self):
        self.vae.model.decode = self.original
        self.pre.remove()
        self.post.remove()

    def before(self, module, args):
        self.chunk_start = self.event()

    def after(self, module, args, output):
        self.chunks.append((self.chunk_start, self.event()))

    def decode(self, *args, **kwargs):
        start = self.event()
        result = self.original(*args, **kwargs)
        self.core.append((start, self.event()))
        return result

    def results(self):
        return dict(vae_core_event_ms=sum(a.elapsed_time(b) for a, b in self.core),
                    vae_chunk_event_ms=[a.elapsed_time(b) for a, b in self.chunks])

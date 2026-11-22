"""Measure the native sequential CFG forwards and UniPC steps on one stream."""


class DiTSteps:
    def __init__(self, torch, model, scheduler):
        self.torch = torch
        self.model = model
        self.scheduler = scheduler
        self.rows = []
        self.forward_index = 0
        self.start = None

    def event(self):
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def install(self):
        self.forward = self.model.forward
        self.step = self.scheduler.step
        self.model.forward = self.forward_timed
        def timed_step(scheduler, *args, **kwargs):
            return self.step_timed(scheduler, *args, **kwargs)

        self.scheduler.step = timed_step

    def restore(self):
        self.model.forward = self.forward
        self.scheduler.step = self.step

    def forward_timed(self, *args, **kwargs):
        if self.forward_index == 0:
            self.start = self.event()
            self.first = self.start
        else:
            self.second = self.event()
        result = self.forward(*args, **kwargs)
        if self.forward_index == 0:
            self.first_end = self.event()
        else:
            self.second_end = self.event()
        self.forward_index += 1
        return result

    def step_timed(self, scheduler, model_output, timestep, *args, **kwargs):
        assert self.forward_index == 2
        begin = self.event()
        result = self.step(scheduler, model_output, timestep, *args, **kwargs)
        end = self.event()
        self.rows.append((timestep, self.start, self.first_end, self.second,
                          self.second_end, begin, end))
        self.forward_index = 0
        return result

    def results(self, sample_id, run_id):
        timesteps = self.torch.stack([row[0] for row in self.rows]).tolist()
        return [dict(sample_id=sample_id, run_id=run_id, step_index=i, timestep=ts,
                     dit_forward_ms=start.elapsed_time(first_end) + second.elapsed_time(second_end),
                     scheduler_ms=begin.elapsed_time(end), total_step_ms=start.elapsed_time(end))
                for i, (ts, (_, start, first_end, second, second_end, begin, end))
                in enumerate(zip(timesteps, self.rows), 1)]

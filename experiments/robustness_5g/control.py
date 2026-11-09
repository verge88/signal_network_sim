from experiments.resilient_5g.control import Control as Previous

class Control(Previous):
    def prepare(self,window,scenario):
        super().prepare(window,scenario)
        if self.root.name=='test' and window%600==80:
            # The parent has really stopped the observer at this same window.
            self.down_until=window+20

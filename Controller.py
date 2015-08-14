class Controller:
    def __init__(self,Horizon=0):
        self.Horizon = Horizon

    def action(self,x,k):
        return 0


class staticGain(Controller):
    def __init__(self,gain=0,*args,**kwargs):
        self.gain = gain
        Controller.__init__(self,*args,**kwargs)

    def action(self,x,k):
        return self.gain*x

# class linearQuadraticRegulator(Controller):
#     def __init__(self,LQSys):

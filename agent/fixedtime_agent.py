from . import BaseAgent
import math


class Fixedtime_Agent(BaseAgent):
    def __init__(self, action_space, signal_plan_file_address, name_prefix, iid, single_inter = 0):
        super().__init__(action_space)
        self.iid = iid
        self.single_inter = single_inter
        self.signal_plan_file_address = signal_plan_file_address
        self.name_prefix = name_prefix
        self.last_action = 0
        self.last_query_index = 0

        if not '/' in self.signal_plan_file_address:
            raise Exception("signal plan address invalid")

        if single_inter:
            self.file_name = self.signal_plan_file_address+self.name_prefix+".txt"
        else:
            self.file_name = self.signal_plan_file_address+self.name_prefix+"_"+self.iid+".txt"

        try:
            with open(self.file_name) as f:
                self.signal_plan_list = f.readlines()
            f.close()
        except:
            raise Exception(self.iid+": signal plan file load failed")

    def get_ob(self):
        return 0

    def get_reward(self):
        return 0

    def get_action(self, world):
        current_time = world.eng.get_current_time()
        int_time = int(current_time)

        if self.single_inter:
            self.last_action = int(self.signal_plan_list[int_time])
            if self.last_action > 0:
                self.last_action -= 1
        else:
            i = self.last_query_index
            while True:
                this_line_list = self.signal_plan_list[i].split(',')
                if float(this_line_list[0]) >= current_time:
                    self.last_query_index = i-1
                    self.last_action = int(float(this_line_list[1][:-1]))
                    break
                else:
                    i+=1
        
        return self.last_action
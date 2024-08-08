import numpy as np
import os




class TrainingInfoPrinter:
    def __init__(
            self, loss_list_dict, is_terminal=False, min_max='max'
    ):
        self.clear_terminal = is_terminal
        self.column_names = list(loss_list_dict)
        if min_max == 'max':
            self.mm_func = max
        else:
            self.mm_func = min
        self.column_names += ['P1', 'P5', 'P10', min_max]
        self.column_width = self.get_column_width()
        self.header = self.make_fixed_header()
        self.row_sep_tail = self.make_row_separator()
    
    def print_info(self,
                   i,
                   data_index,
                   max_data_index,
                   len_data,
                   model_loss_dict,
                   ):
        string_len = 10 + 2*len(str(max_data_index)) + 2*len(str(len_data)) + 5
        self.string_len = string_len
        header = self.make_dynamic_header(
                data_index=data_index, len_data=len_data, i=i, max_data_index=max_data_index
        )
        header += self.header
        row_sep = '-'*(string_len + 1) + self.row_sep_tail
        print(header)
        print(row_sep)
        for model_name, model_dict in model_loss_dict.items():
            row = self.make_row(
                    model_name=model_name, model_dict=model_dict, string_init_len=string_len
            )
            print(row)
            print(row_sep)
    
    def get_column_width(self):
        name_list = self.column_names
        column_width = 8
        for name in name_list:
            if len(name)>column_width:
                column_width = len(name)
        return column_width
    
    def make_row_separator(self):
        c = self.column_width
        num_columns = len(self.column_names)
        row_sep = '+'
        base_col = c*'-'+'+'
        row_sep += base_col * num_columns
        return row_sep
    
    def make_fixed_header(self):
        column_names = self.column_names
        column_width = self.column_width
        header = '|'
        for name in column_names:
            header += int(np.ceil((column_width-len(name))/2))*' ' + name + int(np.floor((column_width-len(name))/2))*' '+'|'
        return header
    
    def make_dynamic_header(self, max_data_index, data_index, len_data, i):
        pd = {1:' '*(len(str(max_data_index))-len(str(data_index)))+f'{data_index}',
              2:max_data_index,
              3:' '*(len(str(len_data))-len(str(i)))+f'{i}',
              4:len_data}
        dynm = f'[{pd[1]}/{pd[2]}][{pd[3]}/{pd[4]}]'
        dynm_len = len(dynm)
        
        # print(dynm_len, self.string_len, dynm_len-self.string_len+9)
        first = ' '*(5-max(int(np.floor((dynm_len-self.string_len+9)/2)),0))
        last = ' '*(5-max(int(np.ceil((dynm_len-self.string_len+9)/2)),0))
        
        return f'{first}{dynm}{last}'
    
    def make_row(self, model_name, model_dict, string_init_len):
        column_width = self.column_width
        loss_list_dict = model_dict['loss'].loss_list_dict
        total_list = model_dict['total_list']
        row_output = ' '*(string_init_len - len(model_name)) + model_name + ' |'
        to_add = [np.nanmean(np.array(lst)) for lst in loss_list_dict.values()]
        if len(total_list) == 0:
            to_add += ['NA', 'NA', 'NA', 'NA']
        elif len(total_list) < 5:
            to_add += [total_list[-1], 'NA', 'NA', round(self.mm_func(total_list), 4)]
        elif len(total_list) < 11:
            to_add += [total_list[-1], np.array(total_list[-5:]).mean(), 'NA', round(self.mm_func(total_list), 4)]
        else:
            to_add += [total_list[-1], np.array(total_list[-5:]).mean(), np.array(total_list[-10:]).mean(), self.mm_func(total_list)]
        for loss in to_add:
            if type(loss) != str:
                if loss > 1:
                    loss = round(loss, 4)
                elif loss > .1:
                    loss = round(loss, 4)
                elif loss > .01:
                    loss = round(loss, 5)
                elif loss > .001:
                    loss = round(loss, 6)
                else:
                    loss = round(loss, 6)
            loss = str(loss)
            if len(loss) >= column_width:
                row_output += loss[:column_width]
            else:
                row_output += ' '*(column_width - len(loss)) + loss
            row_output += '|'
        return row_output


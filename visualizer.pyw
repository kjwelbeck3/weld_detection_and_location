import os
import sys
import tkinter
from tkinter.ttk import Button, Checkbutton, Combobox, Entry, Frame, Label, OptionMenu
from tkinter import BooleanVar, StringVar, filedialog, Text, messagebox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils import get_dataset_list, labels_2_string, parse_textbox_labels, parse_labelfile
from plot_utils import add_staple_patch
from detector import Detector

import point_cloud

class visualizer_app:
    def __init__(self):
        self.root = tkinter.Tk()
        self.root.title('Instance Segmentation Visualizer')
        self.root.geometry( "1100x850" )
        # a = "qwertyuiopasdfghjklzxcvbnm"
        # self.dataset_list = [a[i] for i in range(len(a))]
        self.dataset_list = []
        self.markers = []

        self.create_interface()
        # self.root.protocol('WM_DELETE_WINDOW', self.on_closing)

        self.detector = Detector()

        self.root.mainloop()
        self.dataset_folder = None
        

    def create_interface(self):
        # frame_properties = {}
        
        ## Top Section
        self.load_sec = Frame(self.root)
        self.load_sec.grid(row=0, column=0, columnspan=4, sticky='nw', pady=10)

        ## ---Variables
        self.sample_select = StringVar(self.load_sec)
        self.sample_select.set("Select sample from dataset directory")
        self.sample_select.trace('w', self.load_sample)
        self.load_dir_path = StringVar(self.load_sec)

        ## ---Widgets
        self.load_dir_btn =  Button(self.load_sec, text='Load Dataset Directory...', width=12.5, command=self.load_directory)
        self.sample_dropdown = Combobox(self.load_sec, textvariable=self.sample_select, width=40, values=self.dataset_list, state="readonly")#, postcommand=self.load_sample)
        self.load_dir_label = Label(self.load_sec, textvariable=self.load_dir_path)   
        
        self.load_dir_btn.grid(row=0, column=0, padx=10, pady=5, sticky='nw')
        self.sample_dropdown.grid(row=1, column=0, padx=10, pady=5, sticky='nw')
        self.load_dir_label.grid(row=0, column=1)

        ## Labels Section
        self.labels_viz_sec = Frame(self.root)
        self.labels_viz_sec.grid(row=1, column=0, columnspan=4, sticky='nw', pady=10)

        ## ---Variables
        self.display_labels_sec = BooleanVar(self.labels_viz_sec)
        self.enable_labels_adjust = BooleanVar(self.labels_viz_sec)
        # self.

        ## ---Widgets
        self.labels_viz_checkbutton = Checkbutton(self.labels_viz_sec, variable=self.display_labels_sec, text="Visualize Labels")
        self.labels_viz_checkbutton.grid(row=0, column=0, columnspan=4, sticky='nw', pady=10, padx=10)

        self.labels_fig, self.labels_ax = plt.subplots(1, 1, figsize=(8,3), facecolor='none', tight_layout=True)
        self.labels_ax.set_facecolor(self.get_background_color())
        self.labels_canvas = FigureCanvasTkAgg(self.labels_fig, master=self.labels_viz_sec)
        self.labels_canvas.get_tk_widget().grid(row=1, column=0, padx=10)

        self.labels_sideframe = Frame(self.labels_viz_sec)
        self.labels_sideframe.grid(column=1, row=1, padx=10, pady=10, sticky='nw')

        self.labels_textbox = Text(self.labels_sideframe, height=8, width=30)
        self.labels_textbox.grid(row=0, column=0)
        self.labels_textbox.bind("<Shift_L>", self.get_labels_from_textbox)

        self.labels_adjust_frame = Frame(self.labels_sideframe, width=100)
        self.labels_adjust_frame.grid(row=1, column=0,pady=10, sticky='ne' )

        self.labels_adjust_checkbutton = Checkbutton(self.labels_adjust_frame, variable=self.enable_labels_adjust)
        self.labels_adjust_checkbutton.grid(row=0, column=0, sticky='nw')

        self.labels_adjust_button = Button(self.labels_adjust_frame, text='Adjust Labels', command=self.save_adjusted_label)
        self.labels_adjust_button.grid(column=1, row=0, sticky='nw')

        self.labels_reset_button = Button(self.labels_adjust_frame, text='Reset Labels', width=17)
        self.labels_reset_button.grid(row=1, column=0, columnspan=2,sticky='ne')

        ## Predictions Header Section
        self.preds_header_sec = Frame(self.root)
        self.preds_header_sec.grid(row=2, column=0, columnspan=4, sticky='nw', pady=10)
        
        self.weights_select = StringVar(self.preds_header_sec)
        self.weights_select.set("Select weights from model weights directory")
        self.weights_select.trace('w', self.load_detector)

        self.weights_dir = StringVar(self.preds_header_sec)

        self.load_weights_dir_btn =  Button(self.preds_header_sec, text='Load Model Weights Directory...', width=12.5, command=self.load_model_weights_directory)
        self.load_weights_dir_btn.grid(row=0, column=0, padx=10, pady=5, sticky='nw')
        self.load_weights_dir_label = Label(self.preds_header_sec, textvariable=self.weights_dir)
        self.load_weights_dir_label.grid(row=0, column=1)
        

        self.weights_dropdown = Combobox(self.preds_header_sec, textvariable=self.weights_select, width=40, values=self.dataset_list, state="readonly")#, postcommand=self.load_sample)
        self.weights_dropdown.grid(row=1, column=0, padx=10, pady=5, sticky='nw')
        


        ## Predictions Section
        self.preds_viz_sec = Frame(self.root)
        self.preds_viz_sec.grid(row=3, column=0, columnspan=4,sticky='nw', pady=10)
        self.preds_buttons_sec = Frame(self.preds_viz_sec)
        self.preds_buttons_sec.grid(row=0, column=1, sticky='nw', padx=10)

        ## ---Variables
        

        self.display_preds_sec = BooleanVar(self.preds_viz_sec)
        self.seg_sample_select = StringVar(self.preds_viz_sec)
        self.draw_inst_seg = BooleanVar(self.preds_viz_sec, False)
        self.center_by_centroid = BooleanVar(self.preds_viz_sec)
        self.center_by_icp = BooleanVar(self.preds_viz_sec)
        self.center_by_nn = BooleanVar(self.preds_viz_sec)
        self.semantic_or_instance_select = StringVar(self.preds_viz_sec, "By Instance")

        ## ---Widgets


        self.preds_viz_checkbutton = Checkbutton(self.preds_viz_sec, variable=self.display_preds_sec, text="Visualize Predictions")
        self.preds_viz_checkbutton.grid(row=2, column=0, columnspan=4, sticky='nw',  padx=10)

        self.preds_load_sample_button = Button(self.preds_buttons_sec, text="Load Sample..." , command=self.load_sample_for_segmentation)
        self.preds_load_sample_button.grid(row=0, column=0, sticky='nw', padx=10)
        self.preds_inst_cls_toggle_button = Button(self.preds_buttons_sec, textvariable=self.semantic_or_instance_select, command=self.toggle_task_draw)
        self.preds_inst_cls_toggle_button.grid(row=0, column=1, sticky='nw', padx=10)

        self.preds_fig, self.preds_ax = plt.subplots(1, 1, figsize=(8,3), facecolor='none', tight_layout=True)
        self.preds_ax.set_facecolor(self.get_background_color())
        self.preds_canvas = FigureCanvasTkAgg(self.preds_fig, master=self.preds_viz_sec)
        self.preds_canvas.get_tk_widget().grid(row=1, column=0, padx=10)

        self.preds_sideframe = Frame(self.preds_viz_sec)
        self.preds_sideframe.grid(column=1, row=1, padx=10,pady=5, sticky='nw')

        self.preds_centroid_checkbutton = Checkbutton(self.preds_sideframe, variable=self.center_by_centroid, text="Centroid")
        self.preds_centroid_checkbutton.grid(row=0, column=0, sticky='nw')

        self.preds_centroid_textbox = Text(self.preds_sideframe, height=5, width=30, state='disabled')
        self.preds_centroid_textbox.grid(row=1, column=0)

        self.preds_icp_checkbutton = Checkbutton(self.preds_sideframe, variable=self.center_by_icp, text="ICP")
        self.preds_icp_checkbutton.grid(row=2, column=0, sticky='nw')

        self.preds_icp_textbox = Text(self.preds_sideframe, height=5, width=30, state='disabled')
        self.preds_icp_textbox.grid(row=3, column=0)

        self.preds_nn_checkbutton = Checkbutton(self.preds_sideframe, variable=self.center_by_nn, text="NN")
        self.preds_nn_checkbutton.grid(row=4, column=0, sticky='nw')

        self.preds_nn_textbox = Text(self.preds_sideframe, height=5, width=30, state='disabled')
        self.preds_nn_textbox.grid(row=5, column=0)

    def load_directory(self):
        dir_ = filedialog.askdirectory(title="Select Dataset Directory")
        if dir_:
            print(dir_)
            self.dataset_folder = dir_
            code, res, meta = get_dataset_list(dir_)
            print(res)
            if code != -1:
                print(meta)
                self.root_path = meta['root_path']
                self.scans_path = meta['scans_path']
                self.labels_path = meta["labels_path"]
                self.adjusted_labels_path = meta["adjusted_labels_path"]

                self.dataset_list = res
                self.sample_dropdown["values"] = self.dataset_list
                self.load_dir_label["foreground"] = '#000000'
                self.load_dir_path.set(f"Loaded: {dir_}")
               
            else:
                self.load_dir_label["foreground"] = '#ff0000'
                self.load_dir_path.set(res)
                messagebox.showerror(title="Directory Selection Error", message=res)

    def load_model_weights_directory(self):
        dir_ = filedialog.askdirectory(title="Select Directory with Model Weights")
        if dir_:
            print(dir_)
            self.model_weights_folder = dir_
            self.weights_dir.set("Loaded: " + self.model_weights_folder )

            ## List all the .pt files
            files = os.listdir(dir_)
            weight_files = [file for file in files if file.endswith(".pt")]
            self.weights_dropdown["values"] = weight_files
            

        
    def load_sample(self, *args):
        sample_name = self.sample_select.get()
        self.get_scan()
        self.get_label()

    def load_detector(self):
        print("new_detector")
        print(self.weights_select.get())
        # self.detector = Detector(model_params='./model/param_1212_updated_loss_06_64_192_49_1099')

    def get_scan(self):
        scan_path = ""
        if os.path.isfile(self.scans_path+self.sample_select.get()+".csv"):
            scan_path = self.scans_path+self.sample_select.get()+".csv"
        elif os.path.isfile(self.scans_path+self.sample_select.get()+".bin"):
            scan_path = self.scans_path+self.sample_select.get()+".bin"
        elif os.path.isfile(self.scans_path+self.sample_select.get()+".ply"):
            scan_path = self.scans_path+self.sample_select.get()+".ply"

        if scan_path:

            self.pc = point_cloud.point_cloud()
            self.pc.load(scan_path)
            self.pc.draw_normals(self.labels_fig)
            self.labels_canvas.draw()
        else:
            print("Could not find scan")
            raise ValueError('Could not find a scan as labeled with any of the expected formats [.ply, .bin, .csv].')

    def get_label(self):
        label_path = self.labels_path+self.sample_select.get()+".txt"
        if os.path.isfile(self.adjusted_labels_path+self.sample_select.get()+".txt"):
            label_path = self.adjusted_labels_path+self.sample_select.get()+".txt"
        self.label_dict_original = parse_labelfile(label_path)
        labels_string = labels_2_string(self.label_dict_original["welds"])
        self.labels_textbox.delete("1.0", "end")
        self.labels_textbox.insert("end", labels_string)
        self.draw_labels(self.label_dict_original["welds"])

    def get_labels_from_textbox(self, event):
        print("event")
        print(event)
        textbox_str = self.labels_textbox.get("1.0", "end-1c")
        print("textbox_str")
        print(textbox_str)
        self.label_dict = parse_textbox_labels(textbox_str)
        print("self.label_dict")
        print(self.label_dict)
        if self.label_dict[0] != -1:
            self.draw_labels(self.label_dict[1])

    def draw_labels(self, welds):
        # print("self.markers")
        # print(self.markers)
        # for mk in self.markers:
        #     mk.remove()
        #     self.markers.pop(mk)
        for patch in self.labels_ax.patches:
            patch.remove()
        # for coll in self.labels_ax.collections:
        #     coll.remove()
        for weld in welds:
            print(weld)
            markers = add_staple_patch(self.labels_ax, weld['xloc'], weld['yloc'], weld["yaw"], weld['cls'])
            self.markers.append(markers)
        self.labels_canvas.draw()

    def save_adjusted_label(self):
        if self.label_dict[0] != -1:
            fn = self.adjusted_labels_path+self.sample_select.get()+".txt"
            with open(fn, "w") as adjusted_labelfile:
                adjusted_labelfile.write(labels_2_string(self.label_dict[1]))
            res = f"Saved to: {fn}"
            messagebox.showinfo(title="Saving Adjusted Label", message=res)

        else:
            res = f"Could not parse text box correctly.\n [Error]: {self.label_dict[0]}"
            messagebox.showerror(title="Error saving adjusted label", message=res)

    def load_sample_for_segmentation(self):
        file_ = filedialog.askopenfilename(title="Select Sample for segmentation")
        if file_ and file_.endswith((".csv", ".bin", ".ply")):
            print(f"Selected File: {file_}")
            self.seg_sample_select.set(file_)
            self.segment_cloud()
            self.detector.draw(self.preds_fig, self.draw_inst_seg.get() )
            self.preds_canvas.draw()

        else:
            print("[ERROR]: Did not select .csv or .bin file")

    def segment_cloud(self):
        res = self.detector.process_sample(self.seg_sample_select.get())
        # print("res")
        # print(res)

        self.detector.extract_welds()
        # print("self.detector.welds")
        # print(self.detector.welds)

        self.detector.extract_centroids()
        print("self.detector.weld_centroids")
        print(self.detector.weld_centroids)

    def toggle_task_draw(self):
        state = self.draw_inst_seg.get()
        print(state)
        if state == True:
            self.draw_inst_seg.set(False)
            self.semantic_or_instance_select.set("By Instance")
        else:
            self.draw_inst_seg.set(True)
            self.semantic_or_instance_select.set("By Type")

        
        self.detector.draw(self.preds_fig, self.draw_inst_seg.get() )
        self.preds_canvas.draw()

    def get_centroids(self):
        pass

    
    def enable_labels_sec(self):
        pass

    def enable_predictions_sec(self):
        pass

    def enable_adjust(self):
        pass

    def enable_centroid(self):
        pass

    def enable_icp(self):
        pass

    def enable_nn(self):
        pass

    def get_background_color(self):
        c = self.root['bg']
        if c.startswith('#'):
            # for linux, the hex color is directly returned.
            return c
        # otherwise extract color channel values and convert to hex.
        r,g,b = [x >> 8 for x in self.root.winfo_rgb(c)]
        return f'#{r:02x}{g:02x}{b:02x}'

if __name__ == '__main__':
    app = visualizer_app()

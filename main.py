import os
import sys
if sys.__stdout__ is None or sys.__stderr__ is None:
    os.environ['KIVY_NO_CONSOLELOG'] = '1'
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty
from kivy.lang import Builder
from kivy_garden.draggable import KXDraggableBehavior, KXDroppableBehavior
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
from kivy.config import Config
from kivy.clock import Clock
import numpy as np
from math import e
import dill

Config.set('graphics', 'resizeable', 0)

KV_CODE = '''
<LayerBlock>:
    size_hint: (None, None)
    size: (300,150)
    font_size: 20
    drag_cls: 'test'
    drag_timeout: 0
    canvas.before:
        Color:
            rgba: rgba('#00FF00') if self.is_being_dragged else rgba('#AAAAAA')
        Line:
            rectangle: [*self.pos, *self.size, ]

<Field>:
    size_hint: (None, None) 
    size: (300,150)
    drag_classes: ['test', ]
    canvas.before:
        Line:
            dash_length: 4
            dash_offset: 6
            rectangle: [*self.pos, *self.size]

<CustomPopupNum>:
    size_hint: .5, .5
    auto_dismiss: False
    title: 'Settings'
    BoxLayout:
        text_input: text_input
        popup_button: popup_button
        orientation: 'vertical'
        TextInput:
            input_filter: 'float'
            id: text_input
        Button:
            text: 'Confirm'    
            on_release: root.dismiss()
            id: popup_button
            
<CustomPopupGeneric>:
    size_hint: .5, .5
    auto_dismiss: False
    title: 'Settings'
    BoxLayout:
        text_input: text_input
        popup_button: popup_button
        orientation: 'vertical'
        TextInput:
            id: text_input
        Button:
            text: 'Confirm'    
            on_release: root.dismiss()
            id: popup_button
            
<CustomPopupInput>:
    size_hint: .5, .5
    auto_dismiss: False
    title: 'Settings'
    BoxLayout:
        text_input: text_input
        popup_button: popup_button
        orientation: 'vertical'
        TextInput:
            id: text_input
        Button:
            text: 'Confirm'    
            on_press: app.inference()
            on_release: root.dismiss()
            id: popup_button
 
<GraphPopup>:
    auto_dismiss: False
    title: 'Settings'
    BoxLayout:
        id: box
        pos_hint: {"top":1}
        orientation: 'vertical'
        Button:
            text: 'Confirm'    
            on_release: root.dismiss()
            id: popup_button
            
<ReorderableBoxLayout@KXReorderableBehavior+BoxLayout>:

<InputButton>:
    size_hint: (None, None)
    size: root.width, 50
    text: 'Input'
    font_size: 16
    on_release: app.open_data()

<TrainButton>:
    size_hint: (None, None)
    size: root.width, 50
    text: 'Train'
    font_size: 16
    on_press: self.train()
    
<InferenceButton>:
    size_hint: (None, None)
    size: root.width, 50
    text: 'Inference'
    font_size: 16
    on_release: app.open_inference()
    
<LoadModelButton>:
    size_hint: (None, None)
    size: root.width, 50
    text: 'Load Model'
    font_size: 16
    on_release: app.load_model()

<Alert>:        
    size_hint: (1, None)
    size: root.width, 25
    height: self.minimum_height
    readonly: True
    font_size: 16

BoxLayout:
    orientation: 'horizontal'
    height: self.minimum_height
    width: root.width
    
    ScrollView:
        do_scroll_x: False
        do_scroll_y: True
        
        GridLayout:
            pos_hint: {'center_x': self.minimum_width*0.25}
            height: self.minimum_height
            size: root.width/4, root.height*2
            spacing: 15
            size_hint: (None, None)
            cols:1
            id: fields
            
    ScrollView:
        do_scroll_x: False
        do_scroll_y: True

        ReorderableBoxLayout:
            height: self.minimum_height
            size_hint: (None, None)
            orientation: 'vertical'
            drag_classes: ['test', ]
            spacing: 15
            id: holding
'''

## Defining widgets   
class CustomPopupNum(Popup):
    pass

class CustomPopupGeneric(Popup):
    pass

class CustomPopupInput(Popup):
    pass

class GraphPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        box = self.ids.box
        box.add_widget(FigureCanvasKivyAgg(plt.gcf()), len(self.children))  
        
        
class TrainButton(Button):         
    def train(self, *args):
        app = MainApp.get_running_app()
        app.train_model()
        TrainButton.ran_once = True

class LoadModelButton(Button):
    pass

class InputButton(Button):
    pass

class InferenceButton(Button):
    pass

class Alert(TextInput):
    pass

class Value():
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        
        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,))

        def _backward():
            self.grad = (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,))

        def _backward():
            self.grad = (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        s = 1/(1+e**(-x))
        out = Value(s, (self,))
        
        def _backward():
            self.grad = out.data * (1-out.data) * out.grad
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        t = (e ** x - e ** (-x))/(e ** x + e ** (-x))
        out = Value(t, (self,))
        
        def _backward():
            self.grad = (1 - out.data ** 2) * out.grad
        out._backward = _backward
        
        return out
        
    def backward(self):           
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        
        for v in reversed(topo):
            v._backward()
            

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad})'

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, relu=False, sigmoid=False, tanh=False):
        self.weight = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(np.random.uniform(-1, 1))
        self.relu = relu
        self.sigmoid = sigmoid
        self.tanh = tanh
        
    def __call__(self, x):
        try:
            x = [int(xi) for xi in x]
        except:
            try:
                x = [xi.data for xi in x]
            except:
                x = [x.data]
        act = sum((wi*xi for wi,xi in zip(self.weight, x)), self.bias)
        return act.relu() if self.relu else act.sigmoid() if self.sigmoid else act.tanh() if self.tanh else act
    
    def parameters(self):
        return self.weight + [self.bias]

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class Model(Module):
    def __init__(self, layers):
        self.layers = [
            Layer(layers[i].get_nodes_in(), layers[i].get_nodes_out(), relu=(layers[i+1].get_type() == 'relu'), sigmoid=(layers[i+1].get_type() == 'sigmoid'), tanh=(layers[i+1].get_type() == 'tanh')) if (i < len(layers)-1) else Layer(layers[i].get_nodes_in(), layers[i].get_nodes_out()) for i in range(len(layers)) 
            ]
        self.loss_method = 'mse'
        self.learning_rate = 0.01
    
    def __call__(self, x):
        for layer in self.layers:
            if len(layer.neurons) > 0:
                x = layer(x)
        return x
    
    def train_batch(self, x, targets):
        losses = []
        for i in range(len(x)):
            ## get the predictions by the model
            pred = self([int(xi) for xi in x[i]]) 
            print(f'Prediction: {pred}')
            target = Value(int(targets[i]))
            ## evaluate the loss comparing prediction to target
            if self.loss_method == 'mse':
                loss = (target-pred)**2
            losses.append(loss)      
               
        summed_loss = Value(0)
        for loss in losses:
            summed_loss += loss 
        final_loss = summed_loss/len(losses)
        ## reset gradients to prevent gradient accumulation
        self.zero_grad
        ## backwards pass - calculate gradients of parameters with respect with loss
        final_loss.backward()
        ## updating weights according to their gradients (the derivative with respect to prediction)
        for p in self.parameters():          
            ## backpropagation - goal is minimizing loss
            p.data -= self.learning_rate*p.grad
        print(self.parameters())
        return final_loss.data
        
    def train_loop(self, data_x, data_y, batch_size=10, train_all=True, epochs=100):
        all_losses = []
        if train_all == True:
            for i in range(epochs):
                loss = self.train_batch(data_x, data_y)
                all_losses.append(loss)
                print(f'Epoch {i}: Loss - {loss}')
                #print(f'Layer params: {self.layers[0].parameters()}')
                
        else:
            if batch_size > len(data_x):
                batch_size = len(data_x)
            for _ in range(epochs):
                for i in range(0, len(data_x), batch_size):
                    loss = self.train_batch(data_x[i:i+batch_size], data_y[i:i+batch_size])
                    all_losses.append(loss)
                    print(f'Epoch {i}: Loss - {loss}')
                    
        app = App.get_running_app()
        app.root.ids.fields.children[0].text = 'Finished Training'
        
        plt.plot(all_losses)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        
        p1 = GraphPopup()
        p1.title = 'Training Results'
        p1.open()  
                    
    def save_model(self):
        #try:
            with open('./models/saved_model', 'wb') as filehandler:
                print(f'Model: {self} Layers: {self.layers} Params: {self.parameters()}')
                dill.dump(self, filehandler)          
        #except:
            #app = App.get_running_app()
            #app.root.ids.fields.children[0].text = 'Couldn\'t load model.'
        
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        
        
class Field(KXDroppableBehavior, FloatLayout):   
    instances = []
    ran_once = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Field.instances.append(self)
        FloatLayout.size_hint = None, None
    
    def add_widget(self, widget, *args, **kwargs):
        widget.pos_hint = {'x': 0., 'y': 0.,}           
        return super().add_widget(widget)
        
    def accepts_drag(self, touch, draggable):
        return not self.children
            
    def reset_check(self, *args):
        Field.ran_once = False
    
    def build_model(self):
        layers = []
        for instance in Field.instances:
            try:
                layers.append(instance.children[0])  
            except:
                pass

        print(f'Field Layers: {layers}')
        readable_layers = []
        for i in range(len(layers)):
            readable_layers.append(layers[i].get_flavor_text())
        print(f'Saved layers: {readable_layers}')
        return layers

    def get_layers(self):
        return Field.layers
    
    
class LayerBlock(KXDraggableBehavior, Label):   
    nodes_in_txt = StringProperty('')
    nodes_out_txt = StringProperty('')
    ran_once = False
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        FloatLayout.size_hint = None, None
        self.in_field = False
        self.nodes_in_txt = '0'
        self.nodes_out_txt = '0'
        self.nodes_in = 0
        self.nodes_out = 0
        self.layer_type = self.text
    
    def on_drag_success(self, touch):
        if self.in_field == False:
            app = App.get_running_app()
            for child in app.root.ids.fields.children:
                if isinstance(child, TrainButton):
                    app.root.ids.fields.remove_widget(child)    
            for child in app.root.ids.fields.children:
                if isinstance(child, Alert):
                    app.root.ids.fields.remove_widget(child)   
            for child in app.root.ids.fields.children:
                if isinstance(child, InferenceButton):
                    app.root.ids.fields.remove_widget(child)   
            for child in app.root.ids.fields.children:
                if isinstance(child, LoadModelButton):
                    app.root.ids.fields.remove_widget(child)          
            app.root.ids.fields.add_widget(Field())
            app.root.ids.fields.add_widget(TrainButton())
            app.root.ids.fields.add_widget(LoadModelButton())
            app.root.ids.fields.add_widget(InferenceButton())
            app.root.ids.fields.add_widget(Alert())
            
            self.reset_layers()
            self.in_field = True
        if self.layer_type == 'linear':
            self.set_nodes()
            self.text = f'{self.layer_type} - {self.nodes_in} - {self.nodes_out}'
        if self.layer_type == 'relu':
            self.text = 'relu'
        if self.layer_type == 'sigmoid':
            self.text = 'sigmoid'
        if self.layer_type == 'tanh':
            self.text = 'tanh'
        return super().on_drag_success(touch)
    
    def set_nodes(self):
        if LayerBlock.ran_once == False:
            p1 = CustomPopupNum()
            p1.title = 'Output Shape'
            p1.content.text_input.bind(text=self.setter('nodes_out_txt'))
            p1.content.popup_button.bind(on_press=self.update_nodes)
            p1.open()  
            p2 = CustomPopupNum()
            p2.title = 'Input Shape'
            p2.content.text_input.bind(text=self.setter('nodes_in_txt'))
            p2.content.popup_button.bind(on_press=self.update_nodes)
            p2.open()  
            LayerBlock.ran_once = True
            Clock.schedule_once(self.reset_check)
    
    def update_nodes(self, *args):
        try:
            self.nodes_in = int(self.nodes_in_txt)
        except:
            self.nodes_in = 0
        self.nodes_out = int(self.nodes_out_txt)
        self.text = f'{self.layer_type} - {self.nodes_in} - {self.nodes_out}'
    
    def reset_layers(self):
        app = App.get_running_app()
        app.root.ids.holding.clear_widgets()
        app.root.ids.holding.add_widget(LayerBlock(text=str('linear')))
        app.root.ids.holding.add_widget(LayerBlock(text=str('relu')))
        app.root.ids.holding.add_widget(LayerBlock(text=str('sigmoid')))
        app.root.ids.holding.add_widget(LayerBlock(text=str('tanh')))

    def on_touch_down(self, touch):
        if touch.is_double_tap and self.in_field == True:
            self.set_nodes()
        return super().on_touch_down(touch)

    def reset_check(self, *args):
        LayerBlock.ran_once = False
    
    def get_nodes_in(self):
        return self.nodes_in
    
    def get_nodes_out(self):
        return self.nodes_out
    
    def get_type(self):
        return self.layer_type
    
    def get_flavor_text(self):
        return f'{self.layer_type} - {self.nodes_in_txt} - {self.nodes_out_txt}'
            
            
## Load kivy code
kv = Builder.load_string(KV_CODE)
    
    
class MainApp(App):
    data = StringProperty('')
    input_data = StringProperty('')
    ran_once = False
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = ''
        self.input_data = ''
        self.model = None
    
    def build(self):
        return kv
    
    def on_start(self):
        holding = self.root.ids.holding
        fields = self.root.ids.fields
        fields.add_widget(InputButton())
        fields.add_widget(Field())
        fields.add_widget(TrainButton())
        fields.add_widget(LoadModelButton())
        fields.add_widget(InferenceButton())
        fields.add_widget(Alert(text=' '))
        holding.add_widget(LayerBlock(text=str('linear')))
        holding.add_widget(LayerBlock(text=str('relu')))
        holding.add_widget(LayerBlock(text=str('sigmoid')))
        holding.add_widget(LayerBlock(text=str('tanh')))
        
    
    def train_model(self):
        try:
            self.create_model()
            data_x, data_y, success = self.process_data(self.data)
            if success == False:
                return
            self.model.train_loop(data_x, data_y, train_all=True)
            self.model.save_model()
            self.root.ids.fields.children[0].text = "Successfully Trained!"
        except:
            self.root.ids.fields.children[0].text = "Please enter data first."
        
    def create_model(self, *args):
        layers = self.root.ids.fields.children[-2].build_model()
        self.model = Model(layers)
        
    def load_model(self):
        #try:
            with open('./models/saved_model', 'rb') as loaded_file:
                self.model = dill.load(loaded_file)
            print(f'Model object: {self.model}')
            print(f'Model params: {self.model.parameters()}')
            print(f'Model layers: {self.model.layers}')
            
            self.root.ids.fields.clear_widgets()
            self.root.ids.fields.add_widget(InputButton())
            self.root.ids.fields.add_widget(Field())
            self.root.ids.fields.add_widget(TrainButton())
            self.root.ids.fields.add_widget(LoadModelButton())
            self.root.ids.fields.add_widget(InferenceButton())
            self.root.ids.fields.add_widget(Alert())
            self.root.ids.fields.children[0].text = "Loaded Model!"
        #except:
            #self.root.ids.fields.children[0].text = "No Model Found."
            
        
    def open_data(self):
        if MainApp.ran_once == False:
            p1 = CustomPopupGeneric()
            p1.title = 'Input'
            p1.content.text_input.bind(text=self.setter('data'))
            p1.open()  
            MainApp.ran_once = True
            Clock.schedule_once(self.reset_check)
    
    def process_data(self, input_data):
        try:
            data_line = input_data.split('\n')
            seperated_data = []
            for i in range(len(data_line)):
                seperated_data.append(data_line[i].split('='))
            data_x = []
            data_y = []
            for i in range(len(seperated_data)-1):
                data_x_lists = seperated_data[i][0][1:-1].split(",")
                data_x.append(data_x_lists)
                data_y.append(int(seperated_data[i][1]))
            return data_x, data_y, True
        except Exception as e: 
            print(f'Process data error: {e}')
            self.root.ids.fields.children[0].text = "Please enter data"
            return 0, 0, False
    
    def open_inference(self):
        if MainApp.ran_once == False:
            p1 = CustomPopupInput()
            p1.title = 'Input'
            p1.content.text_input.bind(text=self.setter('input_data'))
            p1.open()  
            MainApp.ran_once = True
            Clock.schedule_once(self.reset_check)
            
    def inference(self):
        model = self.model
        inference_data = self.input_data[1:-1].split(',')
        inference_data = [int(data) for data in inference_data]
        print(f'Model params: {model.parameters()}')
        print(f'Input Data: {inference_data}')
        output = model(inference_data)
        app = App.get_running_app()
        app.root.ids.fields.children[0].text = f'{output.data}'
        
    def reset_check(self, *args):
        MainApp.ran_once = False
        
        
if __name__ == '__main__':
    MainApp().run()
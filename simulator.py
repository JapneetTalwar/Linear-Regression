import torch
from torch import nn

import pygame as pg
import sys,random,math

from Linear_Regressor import LinearRegressor


#colours for ease of use
GREEN = (0,255,0)
BLACK = (0,0,0)
class Node(pg.sprite.Sprite): # datapoint class
    def __init__(self,x,y):
        super().__init__() 
        self.x = x
        self.y = y
        self.pos = pg.Vector2(self.x,self.y)
        self.radius = 5
        
        #normalise nodes so x and y are at 0,0
        self.mlx = self.x - 100
        self.mly = self.y - 10
        
        self.image = pg.Surface((self.radius * 2, self.radius * 2), pg.SRCALPHA)
        pg.draw.circle(self.image, GREEN, (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect(center=(self.x, self.y))
        
    def draw(self,screen):
        self.image = pg.Surface((self.radius * 2, self.radius * 2), pg.SRCALPHA)
        pg.draw.circle(screen, (255,0,0), (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect(center=(self.x, self.y))
    
class Button(pg.sprite.Sprite): # Button class for UI
    def __init__(self, x, y, width, height, text,font_size=40):
        super().__init__()
        self.rect = pg.Rect(x, y, width, height)
        self.text = text
        self.x =x
        self.y = y
        self.width,self.height = width,height
        self.font = pg.font.SysFont(None, font_size)

    def draw(self, screen) :
        pg.draw.rect(screen, GREEN, (self.x, self.y, self.width, self.height), 5)
        
        text_surf = self.font.render(self.text, True, GREEN)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
               
    def is_clicked(self):
        mouse_x, mouse_y = pg.mouse.get_pos()
        return self.rect.collidepoint(mouse_x, mouse_y) and pg.mouse.get_pressed()[0]
    
        
class databox(pg.sprite.Sprite):
    def __init__(self, x, y, width, height,font_size=20):
        super().__init__()
        self.rect = pg.Rect(x, y, width, height)
        self.x =x
        self.y = y
        self.text = None
        self.width,self.height = width,height
        self.font = pg.font.SysFont(None, font_size)
        
    def update(self, epoch, train_loss, test_loss,r2):
        train_loss =  torch.tensor(train_loss, dtype=torch.float16).item()
        test_loss = torch.tensor(test_loss, dtype=torch.float16).item()
        
        self.text = f"Epoch: {epoch}  Train_loss: {train_loss:.3f}  Test_loss: {test_loss:.3f}  R^2 = {r2:.3f}"

    def draw(self, screen) :
        pg.draw.rect(screen, GREEN, (self.x, self.y, self.width, self.height), 5) 
        text_surf = self.font.render(self.text, True, GREEN)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        

class Graph(pg.sprite.Sprite): 
    def __init__(self):
        super().__init__()
        self.width = 440
        self.length = int(240* 1.5)
        
        self.x = 100
        self.y = 10
        
        self.rect = pg.Rect(self.x, self.y, self.width, self.length)
        self.nodes = pg.sprite.Group() 
        
        
        # Calculate the precise center of your box
        self.center_x = self.x + 30   # 320
        self.center_y = self.y + self.length - 30 # 190
        
        self.text = None
    
    def upd_rule(self,model):
        m = model.weights.item() * self.width
        c = model.bias.item() * self.length
        
        self.text = f"y = {m:.3f} x + {c:.3f}"

        
    def draw(self,screen): 
        pg.draw.rect(screen, GREEN, (100, 10, self.width, self.length), 5)    

        # Arguments: (surface, color, start_pos, end_pos, thickness)
        pg.draw.line(screen, GREEN, (self.x+28, self.center_y), (self.x + self.width-2, self.center_y), 5)
        
        pg.draw.line(screen, GREEN, (self.center_x, self.y), (self.center_x, self.y + self.length-30), 5)
        
        font = pg.font.SysFont(None, 20)
        text_surface = font.render(self.text, True, GREEN)

        #Rule
        screen.blit(text_surface, (self.x + self.length -10, self.y+self.width-10)) 
    
    def spawn_node(self,x,y): 
        if x > self.center_x and y < self.center_y:
            self.nodes.add(Node(x,y)) 
    
    def collect_data(self):
        x = []
        y = []

        # graph bottom corner = centre_x + centre_y 
        for node in self.nodes:
            norm_x = (node.x-self.center_x) / float(self.width)
            norm_y = (-node.y+self.center_y) / float(self.length)
            
            x.append([norm_x])
            y.append([norm_y]) # flips it so bottom is where it starts
            
        return torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)

def train_epoch(X_train,Y_train,model,optimiser,loss_fn):
    model.train()
    y_pred = model(X_train)

    
    loss = loss_fn(y_pred,Y_train)
    
    optimiser.zero_grad()
    
    loss.backward()
    
    optimiser.step()
    
    return loss,y_pred

def spawn_border(screen):
        pg.draw.rect(screen, BLACK, (0, 0, 640, 10), 0)  
        #pg.draw.rect(screen, BLACK, (0, 0, 140, 400), 0)  
        pg.draw.rect(screen, BLACK, (540, 0, 250, 480), 0)     
        pg.draw.rect(screen, BLACK, (0, 340, 650, 150), 0) 
        
     
def main():
    pg.init()
    
    #screen settings
    size = width, height = 640,480
    colour = (10, 0, 0) # background colour0, 480 # screen size


    #pygame settings
    screen = pg.display.set_mode(size, pg.SCALED | pg.RESIZABLE)
    clock = pg.time.Clock() 
    fullscreen = False
    
    
    #Graph
    graph = Graph()
    # UI
    pg.display.set_caption("Loading...")
    pg.display.set_icon(pg.image.load('icon.png'))
    
    clear = Button(100,420,100,50,"Clear")
    start = Button(220,420,100,50,"Start")
    
    stop = Button(340,420,100,50,"Stop")
    
    data = databox(100,380,graph.width,30)
    
    #ML code
    model_running = False
    epochs = 20010   
    epoch = 0
    
    #Linear Regression
    model = LinearRegressor()   
    optimiser = torch.optim.SGD(params=model.parameters(), lr= 0.1)
    epoch = 0
    loss_fn = torch.nn.HuberLoss()  
    x, y = None, None
    X_train, Y_train, X_test, Y_test = None, None, None, None
    train_loss_values = []
    test_loss_values = []  
    show_line = False 
    pg.display.set_caption("Linear Regression")  
    
         
    while True:
        screen.fill(colour) # refresh screen every frame
        
        for event in pg.event.get(): 
            if event.type == pg.QUIT: #check if X is pressed to turn off
                pg.quit()    
                sys.exit()
            
            if event.type == pg.MOUSEBUTTONDOWN:
                if not model_running:
                    if graph.rect.collidepoint(pg.mouse.get_pos()): # spawn nodes
                        mx,my = pg.mouse.get_pos() 
                        if event.button == 1:
                            graph.spawn_node(mx,my)
                        elif event.button == 3:
                            for node in graph.nodes:
                                if (node.x - mx)**2 + (node.y -my)**2 <= (node.radius+5)**2:
                                    node.kill()
                                    graph.nodes.remove(node)
                
                    elif clear.is_clicked(): #clear button
                        for sprite in graph.nodes:
                            sprite.kill()
                        show_line = False 
                        data.update(0,0,0,0)
                
                
                    elif start.is_clicked(): #start
                        if len(graph.nodes) >= 5: # ML code
                            model_running = True 
                            x,y = graph.collect_data()
                            show_line = True
                            optimiser = torch.optim.SGD(params=model.parameters(), lr= 0.1)
                elif stop.is_clicked():
                    model_running = False
                    
                    
                elif start.is_clicked():
                    if not model_running:
                        model_running = True
                        
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_f or event.key == pg.K_ESCAPE: 
                    fullscreen = not fullscreen 
                    pg.display.toggle_fullscreen()
        
        #Model
        #ml cooldown
        if model_running:

            
            #80:20 train test split
            train_size = math.floor(x.size(0)*0.8)
            
            X_train = x[:train_size]
            Y_train = y[:train_size]
            X_test = x[train_size:]
            Y_test = y[train_size:]
            
            if epoch < epochs:
                loss,y_pred = train_epoch(X_train,Y_train,model,optimiser,loss_fn)
            
                # Testing code
                #model in evaluation mode 
                model.eval()
                #inference mode
                with torch.inference_mode(): 
                    #forward_pass
                    test_pred = model(X_test)
                    # calc loss

                    test_loss = loss_fn(test_pred, Y_test)
                    
                    train_pixel_error = torch.mean(torch.abs(model(X_train) - Y_train)) * graph.length
                    test_pixel_error = torch.mean(torch.abs(model(X_test) - Y_test)) * graph.length
                    
                if epoch % 10 == 0:  
                    train_loss_values.append(loss)
                    test_loss_values.append(test_loss)

                    y_mean = torch.mean(Y_train)

                    # Sum of squares of residuals
                    ss_res = torch.sum((Y_train - y_pred) ** 2)

                    # Total sum of squares
                    ss_tot = torch.sum((Y_train - y_mean) ** 2)
                    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
                    data.update(epoch,train_pixel_error ,test_pixel_error,r2_score)
                                       
                    graph.upd_rule(model)
                    
                epoch +=1
            elif epoch == epochs:
                    optimiser = None
                    loss_fn = torch.nn.HuberLoss() 
                    x, y = None, None
                    X_train, Y_train, X_test, Y_test = None, None, None, None
                    train_loss_values = []
                    test_loss_values = []  
                    model_running = False
                    model_update = False    
                    epoch = 0           

            
        if show_line: 
            with torch.inference_mode(): 
                # x=0 in model space = center_x on screen
                # x_end = right edge of graph in model space
                x_start = 0.0
                x_end = float(graph.x + graph.width - graph.center_x)
                x_end_norm = x_end / graph.width

                y1_norm = model(torch.tensor([[x_start]])).item()
                y2_norm = model(torch.tensor([[x_end_norm]])).item()
                
                y1 = y1_norm * graph.length
                y2 = y2_norm * graph.length

                # Convert model coords back to screen coords
                # screen_x = model_x + center_x
                # screen_y = center_y - model_y  (y was flipped during collect_data)
                screen_x1 = int(graph.center_x + x_start)
                screen_y1 = int(graph.center_y - y1)
                screen_x2 = int(graph.center_x + x_end)
                screen_y2 = int(graph.center_y - y2)

                pg.draw.line(screen, GREEN, (screen_x1, screen_y1), (screen_x2, screen_y2), 5)
                              
        spawn_border(screen)
        objects = [graph,graph.nodes,clear,start,stop,data]
        for object in objects:
            object.draw(screen)

        
        
        pg.display.flip()  
        clock.tick(60)

if __name__ == "__main__":
    main() 

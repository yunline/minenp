import numpy as np
import sys
import pygame
from pygame.locals import *

np.set_printoptions(precision=2, suppress=True)

size=np.array((640,480))
rate=16

font_path='fusion-pixel-font-monospaced-otf-v2023.02.13/fusion-pixel-monospaced.otf'

direction=('z-','z+','x-','x+','y-','y+')

class Block:
    name='void'
    vertex_template=np.array(#前后左右上下(z-,z+,x-,x+,y-,y+)
        [(1,1,1),(1,1,-1),(-1,1,1),(-1,1,-1),
         (1,-1,1),(1,-1,-1),(-1,-1,1),(-1,-1,-1)],
        dtype=np.float64
    )*0.5

    face=np.array([
        (7,3,1,5),(4,0,2,6),
        (6,2,3,7),(5,1,0,4),
        (6,7,5,4),(0,1,3,2),])
    
    face_n_vector=np.array([
        [0,0,-1],[0,0,1],
        [-1,0,0],[1,0,0],
        [0,-1,0],[0,1,0],],
        dtype=np.float64)

    uv=np.array(
        [[[0,0],[0,1],[1,1],[1,0],],[[0,0],[0,1],[1,1],[1,0],],
         [[0,0],[0,1],[1,1],[1,0],],[[0,0],[0,1],[1,1],[1,0],],
         [[0,0],[0,1],[1,1],[1,0],],[[0,0],[0,1],[1,1],[1,0],],],
        dtype=np.float64
    )*16

    tex_ind=['','','','','','']

    def __init__(self,pos):
        self.pos=pos
        self.hide=[0,0,0,0,0,0]

class BlockGrass(Block):
    name='grass'
    tex_ind=['grass_side',
            'grass_side',
            'grass_side',
            'grass_side',
            'grass_top',
            'grass_bottom']

class BlockStone(Block):
    name='stone'
    tex_ind=['stone',
            'stone',
            'stone',
            'stone',
            'stone',
            'stone']

class Cam:
    @staticmethod        
    def Ry(beta):
        return np.array([
            [np.cos(beta) , 0, np.sin(beta)],
            [0            , 1, 0           ],
            [-np.sin(beta), 0, np.cos(beta)]
            ])
    @staticmethod        
    def Rn(a,n): # 绕n向量旋转角a
        cosine=np.cos(a)
        sine=np.sin(a)
        _cos=1-cosine
        return np.array([
            [n[0]**2*_cos+cosine,n[0]*n[1]*_cos+n[2]*sine,n[0]*n[2]*_cos-n[1]*sine],
            [n[0]*n[1]*_cos-n[2]*sine,n[1]**2*_cos+cosine,n[1]*n[2]*_cos+n[0]*sine],
            [n[0]*n[2]*_cos+n[1]*sine,n[1]*n[2]*_cos-n[0]*sine,n[2]**2*_cos+cosine]
        ])
    
    def __init__(self,fov=90):
        self.fov=fov*np.pi/180
        self.fov_cos=np.cos(self.fov/2)
        self.perspective=(size[0]/2)/np.tan(self.fov/2)

        self.pos=np.array([0,-4,0],dtype=np.float64)
        self.mat=np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]],
            dtype=np.float64)
        self.mat_inv=np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]],
            dtype=np.float64)

        self.front=np.array([0,0,1],dtype=np.float64)
        self.right=np.array([1,0,0],dtype=np.float64)
        self.angle_x=0

    def update(self,ax,ay):
        if np.pi/2>self.angle_x+ax>-np.pi/2:
            self.mat=np.dot(self.mat,self.Rn(ax,self.mat[0]))
            self.angle_x+=ax
        self.mat=np.dot(self.mat,self.Ry(ay))
        self.front=np.dot(self.front,self.Ry(ay))
        self.right=np.dot(self.right,self.Ry(ay))
        self.mat_inv=np.linalg.inv(self.mat)

class Scene:
    blocks={}

    def set_block(self,pos,cls):
        block=cls(pos)
        x,y,z=pos
        self.blocks[(x,y,z)]=block

        if (x,y,z-1) in self.blocks:
            block.hide[0]=1
            self.blocks[(x,y,z-1)].hide[1]=1
        if (x,y,z+1) in self.blocks:
            block.hide[1]=1
            self.blocks[(x,y,z+1)].hide[0]=1
        if (x-1,y,z) in self.blocks:
            block.hide[2]=1
            self.blocks[(x-1,y,z)].hide[3]=1
        if (x+1,y,z) in self.blocks:
            block.hide[3]=1
            self.blocks[(x+1,y,z)].hide[2]=1
        if (x,y-1,z) in self.blocks:
            block.hide[4]=1
            self.blocks[(x,y-1,z)].hide[5]=1
        if (x,y+1,z) in self.blocks:
            block.hide[5]=1
            self.blocks[(x,y+1,z)].hide[4]=1
    
    def del_block(self,pos):
        pos=tuple(pos)
        if pos not in self.blocks:
            return
        del self.blocks[pos]
        x,y,z=pos
        if (x,y,z-1) in self.blocks:
            self.blocks[(x,y,z-1)].hide[1]=0
        if (x,y,z+1) in self.blocks:
            self.blocks[(x,y,z+1)].hide[0]=0
        if (x-1,y,z) in self.blocks:
            self.blocks[(x-1,y,z)].hide[3]=0
        if (x+1,y,z) in self.blocks:
            self.blocks[(x+1,y,z)].hide[2]=0
        if (x,y-1,z) in self.blocks:
            self.blocks[(x,y-1,z)].hide[5]=0
        if (x,y+1,z) in self.blocks:
            self.blocks[(x,y+1,z)].hide[4]=0

class Renderer:
    tex={}
    vertex=[]
    uv=[]
    triangles=[]
    triangle_uv=[]
    triangle_tex=[]
    triangle_data=[]

    triangles_array=np.zeros((0,3,3),dtype=np.float64)
    triangle_uv_array=np.zeros((0,3,2),dtype=np.float64)

    def __init__(self,cam):
        self.cam=cam
        self.load_tex()
        self.init_frame_buf()

        self.looking_at=[None,None]
        self.vertex_tmp=None

        self.sun_vector=np.array([0.707,0.707,0],dtype=np.float64)
        self.gamma_vector=np.array([-0.07,-0.27,0.7],dtype=np.float64)
        # gamma向量 [摄像机因子,太阳因子,零次因子]
    
    def load_tex(self):
        def load(name):
            _sf=pygame.image.load(name)
            return np.array(
                pygame.surfarray.array3d(_sf)
                ,dtype=np.float64)

        self.tex['grass_top']=load('tex/grass_top.png')
        self.tex['grass_side']=load('tex/grass_side.png')
        self.tex['grass_bottom']=load('tex/grass_bottom.png')
        self.tex['stone']=load('tex/stone.png')

    def init_frame_buf(self):
        self.frame_sf=pygame.Surface(size//rate)
        self.frame_buf=pygame.surfarray.pixels3d(self.frame_sf)
    
    def convert_model(self,models): # 方块结构→三角形 
        self.vertex.clear()
        self.uv.clear()
        self.triangles.clear()
        self.triangle_uv.clear()
        self.triangle_tex.clear()
        self.triangle_data.clear()
        for model_pos in models:
            model=models[model_pos]
            self.vertex.extend(model.vertex_template+model.pos)
        self.transform_vertex()

        cnt=0
        for model_pos in models:
            model=models[model_pos]
            for n,(face,tex,hide,nv,uv) in enumerate(zip(
                model.face,model.tex_ind,model.hide,model.face_n_vector,model.uv)):

                brightness_cam=nv.dot(model_pos-0.5*nv-self.cam.pos)

                brightness_sun=nv.dot(self.sun_vector)
                # 亮度越小（负数）材质越亮
                if (not hide) and brightness_cam<-self.cam.fov_cos: # 相邻剔除 背面剔除
                    _f=[self.vertex_tmp[face[i]+cnt] for i in range(4)]
                    t1=[_f[0],_f[1],_f[2]]
                    t2=[_f[0],_f[3],_f[2]]
                    user_data=(tex,model,n,
                        (brightness_cam,brightness_sun,1))

                    if not any(i[2]<0 for i in t1):
                        self.triangles.append(t1)
                        self.triangle_tex.append(self.tex[tex])
                        self.triangle_data.append(user_data)
                        self.triangle_uv.append([uv[0],uv[1],uv[2]])

                    if not any(i[2]<0 for i in t2):
                        self.triangles.append(t2)
                        self.triangle_tex.append(self.tex[tex])
                        self.triangle_data.append(user_data)
                        self.triangle_uv.append([uv[0],uv[3],uv[2]])

            cnt+=8

    def transform_vertex(self):
        if not self.vertex:
            return
        _vertex=np.dot(self.vertex-self.cam.pos,self.cam.mat_inv)
        _vertex[:,2]=1/_vertex[:,2]
        _vertex[:,:2]*=self.cam.perspective*_vertex[:,(2,2)]
        self.vertex_tmp=_vertex+(size[0]/2,size[1]/2,0)

    def draw_wireframe(self,sf):
        self.debug_cnt=0
        for face in self.triangles:
            face=np.array(face)
            l=[]
            for i in face:
                l.append(i[0:2])
            l=np.array(l)
            if np.all(l<0):
                continue
            pygame.draw.aalines(sf,(255,255,255,255),4,l)

    @staticmethod
    def get_array_len(_list):
        return (len(_list)//100+1)*100

    def init_triangles_array(self):
        if len(self.triangles_array)<len(self.triangles):
            self.triangles_array=np.zeros(
                (self.get_array_len(self.triangles)
                ,3,3),dtype=np.float64)
            self.triangles_array[:,1]=[2,0,0]
            self.triangles_array[:,2]=[0,2,0]
    
    def init_triangle_uv_array(self):
        if len(self.triangle_uv_array)<len(self.triangle_uv):
            self.triangle_uv_array=np.zeros(
                (self.get_array_len(self.triangle_uv),
                3,2),dtype=np.float64)
            self.triangle_uv_array[:,1]=[2,0]
            self.triangle_uv_array[:,2]=[0,2]

    def draw_fragment(self):
        self.frame_buf[:,:]=[80,100,220]

        if not self.triangles:
            return

        self.init_triangles_array()
        self.triangles_array[:len(self.triangles)]=self.triangles
        triangles=self.triangles_array

        self.init_triangle_uv_array()
        self.triangle_uv_array[:len(self.triangle_uv)]=self.triangle_uv
        triangle_uv=self.triangle_uv_array

        t_vectors_inv=np.linalg.inv(
            triangles[:,(1,2),:2]-triangles[:,(0,0),:2])
        uv_vectors=triangle_uv[:,(1,2)]-triangle_uv[:,(0,0)]
        delta_z=triangles[:,(1,2),2]-triangles[:,(0,0),2]

        _mn=np.zeros((len(triangles),2),dtype=np.float64)

        def z_buffer(p):
            # Optimized "for i: mn[i].dot(t_vectors_inv[i])"
            _mn[:,0]=p[:,0]*t_vectors_inv[:,0,0]+p[:,1]*t_vectors_inv[:,1,0]
            _mn[:,1]=p[:,0]*t_vectors_inv[:,0,1]+p[:,1]*t_vectors_inv[:,1,1]

            z_tmp,n_tmp,mn_tmp=0,0,[0,0] # z-buffer
            for n,mn in zip(range(len(self.triangles)),_mn):
                if mn[0]+mn[1]<1 and 0<mn[0] and 0<mn[1]:
                    z=mn.dot(delta_z[n])+triangles[n,0,2]

                    if z>z_tmp:
                        z_tmp,n_tmp,mn_tmp=z,n,mn
            return z_tmp,n_tmp,mn_tmp
        
        def sample(uv,tex,border,brightness):
            if border and (any(uv>15) or any(uv<1)):
                return [255,255,255]

            u,v=int(uv[0])%16,int(uv[1])%16
            _b=self.gamma_vector.dot(brightness)
            return [int(i) if i<256 else 255 for i in tex[u,v]*_b]

        # get looking-at
        z_tmp,n_tmp,mn_tmp=z_buffer((size/2)-triangles[:,0,:2])
        if z_tmp>1/4:
            self.looking_at[0]=self.triangle_data[n_tmp][1]
            self.looking_at[1]=self.triangle_data[n_tmp][2]
        else:
            self.looking_at[0]=None
        
        # z-buffer and sampling
        for p in ((x,y) for x in range(0,size[0],rate) for y in range(0,size[1],rate)):
            z_tmp,n_tmp,mn_tmp=z_buffer(p-triangles[:,0,:2])    
            if z_tmp>0:
                uv=(mn_tmp*(triangles[n_tmp,(1,2),2])/(z_tmp)).dot(uv_vectors[n_tmp])
                data=self.triangle_data[n_tmp]
                self.frame_buf[p[0]//rate,p[1]//rate]=sample(uv,self.triangle_tex[n_tmp],
                    data[1] is self.looking_at[0],data[3])

class Player:
    def __init__(self,cam,scene,renderer):
        self.cam=cam
        self.scene=scene
        self.renderer=renderer
        self.acc=np.array([0,0,0],np.float64)
        self.vel=np.array([0,0,0],np.float64)
        self.block_in_hand=BlockStone
        self.floting=1
        self.fly=0

        if self.fly:
            self.vel_max=np.array([5,10,5])
            self.damping_vector=np.array([4,10,4],np.float64)
            self.acc_base=np.array([0,0,0],dtype=np.float64)
        else:
            self.vel_max=np.array([2,10,2])
            self.damping_vector=np.array([8,0,8],np.float64)
            self.acc_base=np.array([0,20,0],dtype=np.float64)
    
    def update_motion(self,dt):
        if all(self.acc==self.acc_base):
            self.vel-=self.vel*(dt*self.damping_vector).clip(0,1)
        self.vel+=self.acc*dt
        self.vel=self.vel.clip(-self.vel_max,self.vel_max)
        self.cam.pos[:]=self.get_collided(
            self.vel*dt,self.cam.pos)
        
        self.acc[:]=self.acc_base
    
    def jump(self):
        if self.fly:
            self.acc[1]-=40
            return
        if not self.floting:
            self.acc[1]-=120
    
    def get_collided(self,delta,src):
        xr,yr,zr=src+delta
        xs,ys,zs=src
        out=[xr,yr,zr]
        self.floting=1
        if any(i in self.scene.blocks for i in ((round(x),round(y),round(z)) 
                                            for y in [yr+1.5,yr-0.5]
                                            for z in [zs+0.2,zs-0.2,zs]
                                            for x in [xs+0.2,xs-0.2,xs]
                                            )):
            if self.vel[1]>0:
                self.floting=0
                out[1]=np.ceil(ys)
            else:
                out[1]=np.floor(ys)
            self.vel[1]=0
        if any(i in self.scene.blocks for i in ((round(x),round(y),round(z)) 
                                            for x in [xr+0.2,xr-0.2,]
                                            for y in [ys-0.2,ys+1.2,ys+0.2]
                                            for z in [zs+0.2,zs-0.2,zs])):
            out[0],self.vel[0]=xs,0
        if any(i in self.scene.blocks for i in ((round(x),round(y),round(z)) 
                                            for z in [zr+0.2,zr-0.2]
                                            for y in [ys-0.2,ys+1.2,ys+0.2]
                                            for x in [xs+0.2,xs-0.2,xs])):
            out[2],self.vel[2]=zs,0

        return out

    def get_target_block(self):
        if self.renderer.looking_at[0] is None:
            return
        
        self.block_in_hand=self.renderer.looking_at[0].__class__

    def put_block(self):
        if self.renderer.looking_at[0] is None:
            return

        pos=self.renderer.looking_at[0].pos+\
            Block.face_n_vector[self.renderer.looking_at[1]]

        _x,_y,_z=self.cam.pos
        if any((*pos,)==i for i in ((round(x),round(y),round(z)) 
                                    for y in [_y,_y+1]
                                    for z in [_z,_z+0.2,_z-0.2]
                                    for x in [_x,_x+0.2,_x-0.2]
                                    )):
            return
        self.scene.set_block(pos,self.block_in_hand)
    
    def dig_block(self):
        if self.renderer.looking_at[0] is None:
            return
        self.scene.del_block(self.renderer.looking_at[0].pos)

class App:
    def __init__(self):
        pygame.font.init()
        self.sf=pygame.display.set_mode(size)
        pygame.display.set_caption('NPMC')
        self.cam=Cam()
        self.scene=Scene()
        self.renderer=Renderer(self.cam)
        self.player=Player(self.cam,self.scene,self.renderer)
        self.timer=pygame.time.Clock()
        self.font=pygame.font.Font(font_path,20)

        pygame.event.set_grab(1)
        pygame.mouse.set_visible(0)
        pygame.key.stop_text_input()
    
    def handle_event(self):
        for event in pygame.event.get():
            if event.type==QUIT:
                self.off=1
            if event.type==MOUSEMOTION:
                self.cam.update(-event.rel[1]/500,-event.rel[0]/500)
            if event.type==MOUSEBUTTONDOWN:
                if event.button==1: # 左键
                    self.player.dig_block()
                if event.button==2:
                    self.player.get_target_block()
                if event.button==3: # 右键
                    self.player.put_block()
                    
            if event.type==KEYDOWN:
                if not self.player.fly and event.key==K_SPACE:
                    self.player.jump()
                    self.player.update_motion(0.07)

        t=self.timer.get_time()
        keys=pygame.key.get_pressed()
        if keys[K_w]:
            self.player.acc+=self.cam.front*40
        if keys[K_s]:
            self.player.acc-=self.cam.front*40
        if keys[K_a]: 
            self.player.acc-=self.cam.right*40
        if keys[K_d]:
            self.player.acc+=self.cam.right*40
        if keys[K_LSHIFT]:
            self.player.acc[1]+=40
        if self.player.fly and keys[K_SPACE]:
            self.player.jump()

        self.player.update_motion(t/1000)

    def draw_debug_info(self):
        border=5
        h,dh=border,self.font.get_height()
        l_at=self.renderer.looking_at
        for text in [
            'Res %d*%d'%(*size//rate,),
            'FPS:%.2f'%self.timer.get_fps(),
            'XYZ:%.2f/%.2f/%.2f'%(*self.cam.pos,),
            'Front:%.2f/%.2f'%(self.cam.front[0],self.cam.front[2]),
            'Vel:%.2f/%.2f/%.2f'%(*self.player.vel,),
            'Floating:%s'%bool(self.player.floting),
            ('Looking-at:None'
                if l_at[0] is None else
             'Looking-at:%d/%d/%d %s'%(*l_at[0].pos,direction[l_at[1]])),
            ('Target:None'
                if l_at[0] is None else
             'Target:%s'%l_at[0].name),
            ]:
            txsf=self.font.render(text,True,(255,255,255),(36,36,36))
            self.sf.blit(txsf,(border,h,1,1),special_flags=BLEND_RGBA_ADD)
            h+=dh
        
        h,dh=border,self.font.get_height()
        for text in [
            'python %d.%d.%d'%(sys.version_info[:3]),
            'pygame%s %s'%('-ce' if pygame.IS_CE else '',pygame.version.ver),
            'numpy %s'%np.version.version
            ]:
            txsf=self.font.render(text,True,(255,255,255),(36,36,36))
            self.sf.blit(txsf,
                (size[0]-border-txsf.get_width(),h,1,1),
                special_flags=BLEND_RGBA_ADD)
            h+=dh

    def mainloop(self):
        self.off=0
        while not self.off:
            self.handle_event()

            self.sf.fill((0,0,0))
            self.renderer.convert_model(self.scene.blocks)
            self.renderer.draw_fragment()
            self.sf.blit(pygame.transform.scale(self.renderer.frame_sf,size),(0,0,1,1))

            #self.renderer.draw_wireframe(self.sf)

            pygame.draw.circle(self.sf,(255,255,0,127),(size[0]/2,size[1]/2),3)

            self.draw_debug_info()
            pygame.display.update()
            self.timer.tick(60)

if __name__=='__main__':
    app=App()
    for z in range(0,6):
        for x in range(0,6):
            for y in range(0,2):
                app.scene.set_block((x,y,z),BlockGrass)
            app.scene.set_block((x,2,z),BlockStone)
            
    app.mainloop()
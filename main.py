import numpy as np
import sys
import pygame
from pygame.locals import *
import os
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

np.set_printoptions(precision=2, suppress=True)

size=np.array((640,480))
rate=10
nthreads=8
maxnquad=5000

font_path='fusion-pixel-font-monospaced-otf-v2023.02.13/fusion-pixel-monospaced.otf'

direction=('z-','z+','x-','x+','y-','y+')
tex_ind={'grass_top':0,'grass_side':1,'grass_bottom':2,'stone':3}
def tn2i(name):
    return tex_ind[name]

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

    tex_ind=[0,0,0,0,0,0]

    def __init__(self,pos):
        self.pos=pos
        self.vertex=self.vertex_template+self.pos
        self.hide=[0,0,0,0,0,0]

class BlockGrass(Block):
    name='grass'
    tex_ind=[tn2i('grass_side'),
            tn2i('grass_side'),
            tn2i('grass_side'),
            tn2i('grass_side'),
            tn2i('grass_top'),
            tn2i('grass_bottom')]

class BlockStone(Block):
    name='stone'
    tex_ind=[tn2i('stone'),
            tn2i('stone'),
            tn2i('stone'),
            tn2i('stone'),
            tn2i('stone'),
            tn2i('stone')]

class Cam:
    render_need_update=1
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
    
    def init_fov(self,fov):
        self.fov=fov*np.pi/180
        self.fov_cos=np.cos(self.fov/2)
        self.perspective=(size[0]/2)/np.tan(self.fov/2)

    def __init__(self,fov=90):
        self.init_fov(fov)        

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
        self.render_need_update=1
        if np.pi/2>self.angle_x+ax>-np.pi/2:
            self.mat=np.dot(self.mat,self.Rn(ax,self.mat[0]))
            self.angle_x+=ax
        self.mat=np.dot(self.mat,self.Ry(ay))
        self.front=np.dot(self.front,self.Ry(ay))
        self.right=np.dot(self.right,self.Ry(ay))
        self.mat_inv=np.linalg.inv(self.mat)

class Scene:
    blocks={}
    render_need_update=1

    def set_block(self,pos,cls):
        self.render_need_update=1
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
        self.render_need_update=1
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

class RendererMP:
    tex={}
    quad_data=[]
    nquads=0
    gamma_vector=np.array([-0.01,-0.27,0.72],dtype=np.float64)
    # gamma向量 [摄像机因子,太阳因子,零次项因子]
    looking_at=[]

    def __init__(self,input_queue,output_queue,shape):
        #print(os.getpid())
        frame_shared_mem=SharedMemory("frame_buf")
        self.frame_buf=np.ndarray(shape=shape,dtype=np.uint8,buffer=frame_shared_mem.buf)
        quads_shared_mem=SharedMemory("quads")
        self.quads=np.ndarray(shape=(maxnquad,3,3),dtype=np.float64,buffer=quads_shared_mem.buf)
        quad_tex_shared_mem=SharedMemory("quad_tex")
        self.quad_tex=np.ndarray(shape=(maxnquad,),dtype=np.uint16,buffer=quad_tex_shared_mem.buf)
        quad_brightness_shared_mem=SharedMemory("quad_brightness")
        self.quad_brightness=np.ndarray(shape=(maxnquad,3),dtype=np.float64,buffer=quad_brightness_shared_mem.buf)
        self.load_tex()
        output_queue.put(None)
        while 1:
            msg=input_queue.get()
            if msg[0]=="stop":
                del self.frame_buf
                frame_shared_mem.close()
                del self.quads
                quads_shared_mem.close()
                del self.quad_tex
                quad_tex_shared_mem.close()
                del self.quad_brightness
                quad_brightness_shared_mem.close()
                break
            elif msg[0]=="looking_at":
                self.looking_at=msg[1]
            elif msg[0]=="nquads":
                self.nquads=msg[1]
            elif msg[0]=="draw":
                self.draw_fragment(msg[1])
                output_queue.put(None)

    def load_tex(self):
        def load(name):
            _sf=pygame.image.load(name)
            return np.array(
                pygame.surfarray.array3d(_sf)
                ,dtype=np.float64)

        self.tex[tn2i('grass_top')]=load('tex/grass_top.png')
        self.tex[tn2i('grass_side')]=load('tex/grass_side.png')
        self.tex[tn2i('grass_bottom')]=load('tex/grass_bottom.png')
        self.tex[tn2i('stone')]=load('tex/stone.png')

    def draw_fragment(self,rect):
        if self.nquads==0:
            return
        
        quads=self.quads[:self.nquads]

        q_vectors_inv=np.linalg.inv(
            quads[:,(1,2),:2]-quads[:,(0,0),:2])
        delta_z=quads[:,(1,2),2]-quads[:,(0,0),2]

        _mn=np.zeros((len(quads),2),dtype=np.float64)
        _z=np.ones((len(quads),1),dtype=np.float64)
        _uv=np.zeros((len(quads),2),dtype=np.float64)

        def z_buffer(p):
            # Optimized "for i: mn[i].dot(t_vectors_inv[i])"
            _mn[:,0]=p[:,0]*q_vectors_inv[:,0,0]+p[:,1]*q_vectors_inv[:,1,0]
            _mn[:,1]=p[:,0]*q_vectors_inv[:,0,1]+p[:,1]*q_vectors_inv[:,1,1]

            _z[:,0]=_mn[:,0]*delta_z[:,0]+_mn[:,1]*delta_z[:,1]+quads[:,0,2]

            _uv[:]=_mn*quads[:,(1,2),2]/_z

            z_tmp,n_tmp,uv_tmp=0,0,[0,0] # z-buffer
            for n,z,uv in zip(range(len(quads)),_z,_uv):
                if 0<uv[0]<1 and 0<uv[1]<1:
                    if z>z_tmp:
                        z_tmp,n_tmp,uv_tmp=z,n,uv
            return z_tmp,n_tmp,uv_tmp
        
        def sample(uv,tex,border,brightness):
            if border and (any(uv>15) or any(uv<1)):
                return [255,255,255]
            u,v=int(uv[0]),int(uv[1])
            _b=self.gamma_vector.dot(brightness)
            return [int(min(i,255)) for i in tex[u,v]*_b]
        
        # z-buffer and sampling
        for p in ((x,y) for x in range(rect.left,rect.right,rate) for y in range(rect.top,rect.bottom,rate)):
            z_tmp,n_tmp,uv_tmp=z_buffer(p-quads[:,0,:2]-size/2)    
            if z_tmp>0:
                self.frame_buf[p[0]//rate,p[1]//rate]=\
                    sample(uv_tmp*16,self.tex[self.quad_tex[n_tmp]],
                    n_tmp in self.looking_at,self.quad_brightness[n_tmp])

class Renderer:
    vertex=[]
    quad_data=[]
    looking_at=[None,None]
    looking_at_quads=[]

    def __init__(self,cam,scene):
        self.cam=cam
        self.scene=scene

        self.vertex_tmp=None

        self.sun_vector=np.array([0.707,0.707,0],dtype=np.float64)

        self.nquads=0

        self.frame_sf=pygame.Surface(size//rate)
        self.frame_buf=pygame.surfarray.pixels3d(self.frame_sf)

        self.frame_shared_mem=SharedMemory("frame_buf",create=True,size=self.frame_buf.nbytes)
        self.frame_shared_buf=np.ndarray(shape=self.frame_buf.shape,dtype=np.uint8,buffer=self.frame_shared_mem.buf)

        self.quads_shared_mem=SharedMemory("quads",create=True,size=(maxnquad*3*3*8))
        self.quads=np.ndarray(shape=(maxnquad,3,3),dtype=np.float64,buffer=self.quads_shared_mem.buf)

        self.quad_tex_shared_mem=SharedMemory("quad_tex",create=True,size=(maxnquad*2))
        self.quad_tex=np.ndarray(shape=(maxnquad,),dtype=np.uint16,buffer=self.quad_tex_shared_mem.buf)

        self.quad_brightness_shared_mem=SharedMemory("quad_brightness",create=True,size=(maxnquad*3*8))
        self.quad_brightness=np.ndarray(shape=(maxnquad,3),dtype=np.float64,buffer=self.quad_brightness_shared_mem.buf)

        self.o_queues=[multiprocessing.Queue() for _ in range(nthreads)]
        self.i_queues=[multiprocessing.Queue() for _ in range(nthreads)]
        self.mp=[]
        for i,o in zip(self.i_queues,self.o_queues):
            self.mp.append(multiprocessing.Process(target=RendererMP,args=(o,i,self.frame_buf.shape)))
            self.mp[-1].start()

        for q in self.i_queues:
            q.get()

    def close(self):
        for q in self.o_queues:
            q.put(("stop",))

        for i in self.mp:
            i.join() # 等待进程回收
        
        del self.frame_shared_buf
        self.frame_shared_mem.unlink()
        del self.quads
        self.quads_shared_mem.unlink()
        del self.quad_tex
        self.quad_tex_shared_mem.unlink()
        del self.quad_brightness
        self.quad_brightness_shared_mem.unlink()
    
    def sync_data(self):
        for q in self.o_queues:
            q.put(("looking_at",self.looking_at_quads))
            q.put(("nquads",self.nquads))
    
    def convert_model(self,models):
        if self.cam.render_need_update==0:
            return
        self.cam.render_need_update=0
        self.quad_data.clear()
        self.looking_at_quads.clear()
        if self.scene.render_need_update:
            self.scene.render_need_update=0
            self.vertex=[]
            for model_pos in models:
                self.vertex.extend(models[model_pos].vertex)
        self.transform_vertex()

        cnt=-8
        self.nquads=0
        model_pos_iter=models.keys()
        relative_pos_array=np.array(list(model_pos_iter),dtype=np.float64)-self.cam.pos
        for model_pos,relative_pos in zip(model_pos_iter,relative_pos_array):
            cnt+=8
            model=models[model_pos]
            if all(model.hide): # 跳过完全遮挡
                continue
            r=np.linalg.norm(relative_pos)
            if r>20: # 跳过远距离方块
                continue
            if (relative_pos.dot(self.cam.mat[2])<0 and r>2): # 跳过背面
                continue
            for face_direction,face,tex,hide,nv in zip([0,1,2,3,4,5],
                model.face,model.tex_ind,model.hide,model.face_n_vector):

                brightness_cam=nv.dot(relative_pos-0.5*nv)

                # 亮度越小（负数）材质越亮
                if (not hide) and brightness_cam<-self.cam.fov_cos: # 相邻剔除 背面剔除
                    q=self.vertex_tmp[(face[0]+cnt,face[3]+cnt,face[1]+cnt),]
                    brightness_sun=nv.dot(self.sun_vector)

                    self.quads[self.nquads]=q
                    self.quad_tex[self.nquads]=tex
                    if model is self.looking_at[0]:
                        self.looking_at_quads.append(self.nquads)
                    self.quad_brightness[self.nquads]=(brightness_cam,brightness_sun,1)
                    self.quad_data.append((model,face_direction))
                    self.nquads+=1

        self.sync_data()

    def transform_vertex(self):
        if len(self.vertex)==0:
            return
        _vertex=np.dot(self.vertex-self.cam.pos,self.cam.mat_inv)
        _vertex[:,2]=1/_vertex[:,2]
        _vertex[:,:2]*=self.cam.perspective*_vertex[:,(2,2)]
        self.vertex_tmp=_vertex
    
    def draw_fragment(self):
        self.frame_shared_buf[:]=[80,100,220]
        for n,q in enumerate(self.o_queues):
            q.put(("draw",pygame.Rect(0,n*size[1]//nthreads,
                size[0],size[1]//nthreads)))
        for q in self.i_queues: # wait for render
            q.get()
        self.frame_buf[:]=self.frame_shared_buf

    def update_looking_at(self):
        quads=self.quads[:self.nquads]

        q_vectors_inv=np.linalg.inv(
            quads[:,(1,2),:2]-quads[:,(0,0),:2])
        delta_z=quads[:,(1,2),2]-quads[:,(0,0),2]

        _mn=np.zeros((len(quads),2),dtype=np.float64)
        _z=np.zeros((len(quads),1),dtype=np.float64)
        _uv=np.zeros((len(quads),2),dtype=np.float64)

        p=0-quads[:,0,:2]
        # Optimized "for i: mn[i].dot(t_vectors_inv[i])"
        _mn[:,0]=p[:,0]*q_vectors_inv[:,0,0]+p[:,1]*q_vectors_inv[:,1,0]
        _mn[:,1]=p[:,0]*q_vectors_inv[:,0,1]+p[:,1]*q_vectors_inv[:,1,1]
        _z[:,0]=_mn[:,0]*delta_z[:,0]+_mn[:,1]*delta_z[:,1]+quads[:,0,2]
        _uv[:]=_mn*quads[:,(1,2),2]/_z
        z_tmp,n_tmp,uv_tmp=0,0,[0,0] # z-buffer
        for n,z,uv in zip(range(len(quads)),_z,_uv):
            if 0<uv[0]<1 and 0<uv[1]<1:
                if z>z_tmp:
                    z_tmp,n_tmp,uv_tmp=z,n,uv
        
        # get looking-at
        if z_tmp>1/4:
            self.looking_at[0]=self.quad_data[n_tmp][0]
            self.looking_at[1]=self.quad_data[n_tmp][1]
        else:
            self.looking_at[0]=None

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
            self.vel_max=[5,10]
            self.damping_vector=np.array([4,10,4],np.float64)
            self.acc_base=np.array([0,0,0],dtype=np.float64)
        else:
            self.vel_max=[4,10]
            self.damping_vector=np.array([8,0,8],np.float64)
            self.acc_base=np.array([0,20,0],dtype=np.float64)
    
    def update_motion(self,dt):
        if all(self.acc==self.acc_base):
            self.vel-=self.vel*(dt*self.damping_vector).clip(0,1)
        self.vel+=self.acc*dt
        vxz=self.vel[0]**2+self.vel[2]**2
        if vxz>self.vel_max[0]**2:
            v_rate=self.vel_max[0]/(vxz**0.5)
            self.vel[0]*=v_rate
            self.vel[2]*=v_rate
        self.vel[1]=np.clip(self.vel[1],-self.vel_max[1],self.vel_max[1])
        
        pos=self.get_collided(
            self.vel*dt,self.cam.pos)
        if not all(self.cam.pos==pos):
            self.cam.render_need_update=1
        self.cam.pos[:]=pos
        
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
        self.renderer=Renderer(self.cam,self.scene)
        self.player=Player(self.cam,self.scene,self.renderer)
        self.timer=pygame.time.Clock()
        self.font=pygame.font.Font(font_path,20)

        pygame.event.set_grab(1)
        pygame.mouse.set_visible(0)
        pygame.key.stop_text_input()
        pygame.key.set_repeat(180)
    
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
            'nQuads:%d'%self.renderer.nquads,
            'nThreads:%d'%nthreads,
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
            self.renderer.convert_model(self.scene.blocks)
            self.renderer.update_looking_at()
            self.renderer.draw_fragment()
            self.sf.blit(pygame.transform.scale(self.renderer.frame_sf,size),(0,0,1,1))

            pygame.draw.circle(self.sf,(255,255,0,127),(size[0]/2,size[1]/2),3)

            self.draw_debug_info()
            pygame.display.update()
            self.timer.tick(60)
        self.renderer.close()

if __name__=='__main__':
    multiprocessing.freeze_support()
    app=App()
    for z in range(0,6):
        for x in range(0,6):
            for y in range(0,2):
                app.scene.set_block((x,y,z),BlockGrass)
            app.scene.set_block((x,2,z),BlockStone)
            
    app.mainloop()
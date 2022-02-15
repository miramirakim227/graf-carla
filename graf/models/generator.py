import numpy as np
import torch
from ..utils import sample_on_sphere, look_at, to_sphere
from ..transforms import FullRaySampler
from submodules.nerf_pytorch.run_nerf_mod import render, run_network            # import conditional render
from functools import partial
from .resnet import resnet18


class Generator(object):
    def __init__(self, H, W, focal, radius, ray_sampler, render_kwargs_train, render_kwargs_test, parameters, named_parameters,
                 range_u=(0,1), range_v=(0.01,0.49), chunk=None, device='cuda', orthographic=False):
        self.device = device
        self.H = int(H)
        self.W = int(W)
        self.focal = focal
        self.radius = radius
        self.range_u = range_u
        self.range_v = range_v
        self.chunk = chunk
        coords = torch.from_numpy(np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), -1))
        self.coords = coords.view(-1, 2)

        self.ray_sampler = ray_sampler
        self.val_ray_sampler = FullRaySampler(orthographic=orthographic)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.initial_raw_noise_std = self.render_kwargs_train['raw_noise_std']
        self._parameters = parameters
        self._named_parameters = named_parameters
        # module_dict에 network_fn, network_fine 넣어주는 중 -> discriminator는..?
        self.module_dict = {'generator': self.render_kwargs_train['network_fn']}
        for name, module in [('generator_fine', self.render_kwargs_train['network_fine'])]:
            if module is not None:
                self.module_dict[name] = module
        
        for k, v in self.module_dict.items():
            if k in ['generator', 'generator_fine']:
                continue       # parameters already included
            self._parameters += list(v.parameters())
            self._named_parameters += list(v.named_parameters())
        
        self.parameters = lambda: self._parameters           # save as function to enable calling model.parameters()
        self.named_parameters = lambda: self._named_parameters           # save as function to enable calling model.named_parameters()
        self.use_test_kwargs = False

        self.render = partial(render, H=self.H, W=self.W, focal=self.focal, chunk=self.chunk)
        self.encoder = resnet18(pretrained=True, shape_dim=self.z_dim, app_dim=self.z_dim)

    def __call__(self, z, y=None, uv = uv, rays=None):
        bs = z.shape[0]

        # sample_ray를 uv 기준으로 함..!
        if rays is None:
            rays, selected_idcs, _ = torch.cat([self.sample_rays(uv) for _ in range(bs)], dim=1)        # ray를 하나씩 샘플링

        render_kwargs = self.render_kwargs_test if self.use_test_kwargs else self.render_kwargs_train
        render_kwargs = dict(render_kwargs)        # copy
    
        # in the case of a variable radius
        # we need to adjust near and far plane for the rays
        # so they stay within the bounds defined wrt. maximal radius
        # otherwise each camera samples within its own near/far plane (relative to this camera's radius)
        # instead of the absolute value (relative to maximum camera radius)
        if isinstance(self.radius, tuple):
            assert self.radius[1] - self.radius[0] <= render_kwargs['near'], 'Your smallest radius lies behind your near plane!'
    
            rays_radius = rays[0].norm(dim=-1)
            shift = (self.radius[1] - rays_radius).view(-1, 1).float()      # reshape s.t. shape matches required shape in run_nerf
            render_kwargs['near'] = render_kwargs['near'] - shift
            render_kwargs['far'] = render_kwargs['far'] - shift
            assert (render_kwargs['near'] >= 0).all() and (render_kwargs['far'] >= 0).all(), \
                (rays_radius.min(), rays_radius.max(), shift.min(), shift.max())
            

        # mira: 여기서 rays를 pose로부터 뽑을 수 있도록 하고 <- 그 위에 있는 pose를 찾기 
        # mira: render 여기에 우리의 encoder 넣어줘도 괜찮음! 
        # encoder output으로 camera, features 넣기 
        # self.render를 찾아 나서기!

        # image encoder 넣기! 
        # latent shape, appearance는 어디있음..?
        # mira: 아... features sampling이 z로 이미 들어가네.... 오... 그러면 render kwargs에 dimension 맞춰서 넣어주면 될듯..
        render_kwargs['features'] = z   # B, 256 -> minus plus 골고루 들어감스트 
        
        # mira: 여기에 render가 있음! 여기에 camera 넣으면 됨!
        rgb, disp, acc, extras = render(self.H, self.W, self.focal, chunk=self.chunk, rays=rays, 
                                        **render_kwargs)

        rays_to_output = lambda x: x.view(len(x), -1) * 2 - 1      # (BxN_samples)xC
    
        if self.use_test_kwargs:               # return all outputs
            return rays_to_output(rgb), \
                   rays_to_output(disp), \
                   rays_to_output(acc), extras

        rgb = rays_to_output(rgb)
        return rgb

    def decrease_nerf_noise(self, it):
        end_it = 5000
        if it < end_it:
            noise_std = self.initial_raw_noise_std - self.initial_raw_noise_std/end_it * it
            self.render_kwargs_train['raw_noise_std'] = noise_std

    def sample_pose(self):
        # sample location on unit sphere
        loc = sample_on_sphere(self.range_u, self.range_v)
        
        # sample radius if necessary
        radius = self.radius
        if isinstance(radius, tuple):
            radius = np.random.uniform(*radius)

        loc = loc * radius
        R = look_at(loc)[0]

        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = torch.Tensor(RT.astype(np.float32))
        # mira: check camera pose translation, rotation 확인 
        # mira: check 여기선 radius를 조정하는데.. 우리도 그렇게 해도 되는지 아니면 데이터마다 지금처럼 radius와 그에 따른 near, far를 각각 설정해줘야 하는지
        # mira: 그러면 우리도.. 아예 loc을 예측하자 
        return RT

    def sample_rays(self):  # pose도 input argument로 넣어주기. dimension 맞춰주기!
        # mira: expect uv, range는 dataset의 uv에 맞게 정해져있음 
        pose = self.sample_pose()   # mira: 여기서 pose distribution 찾아보기, 적어도 pose어떻게 생겼는지 확인해보기
        # mira: 그리고 pose dimension도 확인해보기
        sampler = self.val_ray_sampler if self.use_test_kwargs else self.ray_sampler
        batch_rays, _, _ = sampler(self.H, self.W, self.focal, pose)
        return batch_rays

    def to(self, device):
        self.render_kwargs_train['network_fn'].to(device)
        if self.render_kwargs_train['network_fine'] is not None:
            self.render_kwargs_train['network_fine'].to(device)
        self.device = device
        return self

    def train(self):
        self.use_test_kwargs = False
        self.render_kwargs_train['network_fn'].train()
        if self.render_kwargs_train['network_fine'] is not None:
            self.render_kwargs_train['network_fine'].train()

    def eval(self):
        self.use_test_kwargs = True
        self.render_kwargs_train['network_fn'].eval()
        if self.render_kwargs_train['network_fine'] is not None:
            self.render_kwargs_train['network_fine'].eval()

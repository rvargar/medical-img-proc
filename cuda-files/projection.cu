__device__ float3 calcIntersect( float3 dir, float3 TSD, float dz)
{
	float3 det;

	float dist =  (dz-TSD.z)/(dir.z);

	det.x = TSD.x+dir.x*dist;
	det.y = TSD.y+dir.y*dist;
	det.z = TSD.z+dir.z*dist;
	return det;

}


// bilinear interpolation
__device__ float bilinear_interpolate(float x, float y,float a, float b, float c, float d)
{
	float res = a*(1-x)*(1-y) + b*x*(1-y) + c*(1-x)*y + d*x*y;
	return res;

}

//calc distance from integer pixel values
__device__ float calc_d(float number)
{
	float res = floor(number);
	return number-res;
}
inline __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float length(float3 v)
{
   return sqrtf(dot(v, v));
}

__device__ inline float sqr(float a)
{
	return a*a;
}

inline __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ void calcLengths(float*  c, float l, float3 dir, float xv, float yv)
{	int tmpx = 0;
	int tmpy = 0;
	int ind = 1;
	int k=0;
	for(int i=0;i<=20;i++)
	{
		if(int((dir.x*(float(i)*l/20.0f))/xv) == tmpx &&  int((dir.y*(float(i)*l/20.0f))/yv) == tmpy)
		{
			k++;
		}
		else
		{
			c[ind] = float(k)*l/20.f;
			tmpx = int((dir.x*(float(i)*l/20.0f))/xv);
			tmpy = int((dir.y*(float(i)*l/20.0f))/yv);
			c[ind+1]= tmpx;
			c[ind+2]= tmpy;
			ind+=3;
			k = 1;
		}
	}
	if(ind==1)
	{
		c[1] = float(k)*l/20.f;
	}
		c[0] = ind;
}

__device__ float calculateAttenuation(float energy, float material_density, float path_length) {
    // Simplified attenuation model (you should replace with proper material-specific coefficients)
    // This is a placeholder - use proper mass attenuation coefficients for your materials
    float mu = 0.2f * material_density; // cm²/g - example value for water at ~50 keV
    return expf(-mu * material_density * path_length);
}

__device__ float applyBeamHardening(float initial_intensity, float* path_lengths, float* densities, int num_segments) {
    // Simulate polychromatic beam with 3 energy bins (simplified)
    float energies[3] = {30.0f, 60.0f, 90.0f}; // keV
    float weights[3] = {0.3f, 0.4f, 0.3f};     // Relative weights

    float total_transmission = 0.0f;

    for (int e = 0; e < 3; e++) {
        float transmission = 1.0f;
        for (int i = 0; i < num_segments; i++) {
            transmission *= calculateAttenuation(energies[e], densities[i], path_lengths[i]);
        }
        total_transmission += weights[e] * transmission;
    }

    return initial_intensity * total_transmission;
}

//Projection onto arbitrary plane, which is parallel to the x-y plane, and the source and the detector are moving in x direction during the exposure.
__global__ void projection( float* result, float* input,
float* xs, float* ys,float* zs,
float xv, float yv, float zv,
int vx, int vy, int vz,
float* TSD,float* TDD,
float su,float sv,
float nu,float nv,
float* source_coord,
float* detector_coord,
float* coordz , int mode,
cudaTextureObject_t volTex)
{

	unsigned short x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	unsigned short yy = y%int(nv);
	unsigned short z = floor(float(y)/float(nv));
	if(x<nu && yy<nv){
		float3 source = make_float3(source_coord[z],TSD[1],TSD[2]);
		float3 dir = make_float3(source_coord[z]-(detector_coord[z])-((x-nu/2)*su/nu), TSD[1]-TDD[1]-((yy-nv/2)*sv/nv), TSD[2]-TDD[2]);

		float l = sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
		dir.x /= l;
		dir.y /= l;
		dir.z /= l;

		float sum = 0.0f;

	switch(mode)
	{
		case 0: // Nearest Neighbor interpolation
		{
			for(int i = 0; i < vz; i++)
			{
				float3 newdir = calcIntersect(dir, source, coordz[i] + zv/2.0f);
				float3 newdir2 = calcIntersect(dir, source, coordz[i] - zv/2.0f);
				l = length(newdir - newdir2);

				int vol_size = vx * vy * vz;
				int ind1 = int((newdir.x - xs[0])/xv) + int((newdir.y - ys[0])/yv)*(vx) + i*(vy*vx);
				int ind2 = int((newdir2.x - xs[0])/xv) + int((newdir2.y - ys[0])/yv)*(vx) + i*(vy*vx);

				// Clamp to valid range
				ind1 = max(0, min(ind1, vol_size - 1));
				ind2 = max(0, min(ind2, vol_size - 1));

				if((newdir.x >= xs[0] && newdir.x < xs[1]) && (newdir.y >= ys[0] && newdir.y < ys[1]))
				{
					if(abs(newdir.x - newdir2.x) <= xv && abs(newdir.y - newdir2.y) <= yv)
					{
						sum += l * input[ind1];
					}
					else
					{
						// Only use ind2 when newdir2 is also inside the volume
						float contrib2 = 0.0f;
						if((newdir2.x >= xs[0] && newdir2.x < xs[1]) && (newdir2.y >= ys[0] && newdir2.y < ys[1]))
							contrib2 = input[ind2];

						sum += sqrtf(sqr(newdir.x - xv*floor(newdir.x/xv)) + sqr(newdir.y - yv*floor(newdir.y/yv))) * input[ind1] +
						        (l - sqrtf(sqr(newdir.x - xv*floor(newdir.x/xv)) + sqr(newdir.y - yv*floor(newdir.y/yv)))) * contrib2;
					}
				}
			}
			break;
		}
		case 1: // Bilinear interpolation
		{
			for(int i = 0; i < vz; i++)
			{
				float3 newdir = calcIntersect(dir, source, coordz[i] + zv/2.0f);
				float3 newdir2 = calcIntersect(dir, source, coordz[i] - zv/2.0f);
				l = length(newdir - newdir2);

				int vol_size = vx * vy * vz;

				int ind01 = int((newdir.x - xv/2 - xs[0])/xv) + int((newdir.y - yv/2 - ys[0])/yv)*(vx) + i*(vy*vx);
				int ind02 = int((newdir.x + xv/2 - xs[0])/xv) + int((newdir.y - yv/2 - ys[0])/yv)*(vx) + i*(vy*vx);
				int ind03 = int((newdir.x - xv/2 - xs[0])/xv) + int((newdir.y + yv/2 - ys[0])/yv)*(vx) + i*(vy*vx);
				int ind04 = int((newdir.x + xv/2 - xs[0])/xv) + int((newdir.y + yv/2 - ys[0])/yv)*(vx) + i*(vy*vx);

				int ind05 = int((newdir2.x - xv/2 - xs[0])/xv) + int((newdir2.y - yv/2 - ys[0])/yv)*(vx) + i*(vy*vx);
				int ind06 = int((newdir2.x + xv/2 - xs[0])/xv) + int((newdir2.y - yv/2 - ys[0])/yv)*(vx) + i*(vy*vx);
				int ind07 = int((newdir2.x - xv/2 - xs[0])/xv) + int((newdir2.y + yv/2 - ys[0])/yv)*(vx) + i*(vy*vx);
				int ind08 = int((newdir2.x + xv/2 - xs[0])/xv) + int((newdir2.y + yv/2 - ys[0])/yv)*(vx) + i*(vy*vx);

				// Clamp all indices to valid range — guards against borderline float
				// rounding that can push an index one step outside the buffer.
				ind01 = max(0, min(ind01, vol_size - 1));
				ind02 = max(0, min(ind02, vol_size - 1));
				ind03 = max(0, min(ind03, vol_size - 1));
				ind04 = max(0, min(ind04, vol_size - 1));
				ind05 = max(0, min(ind05, vol_size - 1));
				ind06 = max(0, min(ind06, vol_size - 1));
				ind07 = max(0, min(ind07, vol_size - 1));
				ind08 = max(0, min(ind08, vol_size - 1));

				if((newdir.x >= xs[0] && newdir.x < xs[1]) && (newdir.y >= ys[0] && newdir.y < ys[1]))
				{
					// Trilinear interpolation
					float X = calc_d(newdir.x);
					float Y = calc_d(newdir.y);

					float a = bilinear_interpolate(X, Y,
						input[ind01],
						input[ind02],
						input[ind03],
						input[ind04]);

					// Only use newdir2 sample when it is also inside the volume
					float b = 0.0f;
					if((newdir2.x >= xs[0] && newdir2.x < xs[1]) && (newdir2.y >= ys[0] && newdir2.y < ys[1]))
					{
						X = calc_d(newdir2.x);
						Y = calc_d(newdir2.y);
						b = bilinear_interpolate(X, Y,
							input[ind05],
							input[ind06],
							input[ind07],
							input[ind08]);
					}

					if(abs(newdir.x - newdir2.x) <= xv && abs(newdir.y - newdir2.y) <= yv)
					{
						sum += l * ((a + b) / 2.0f);
					}
					else
					{
						sum += sqrtf(sqr(newdir.x - xv*floor(newdir.x/xv)) + sqr(newdir.y - yv*floor(newdir.y/yv))) * a +
						        (l - sqrtf(sqr(newdir.x - xv*floor(newdir.x/xv)) + sqr(newdir.y - yv*floor(newdir.y/yv)))) * b;
					}
				}
			}
			break;
		}
		case 2: // Hardware trilinear interpolation via 3D texture memory
		{
			// Volume is bound as a 3D texture with non-normalised coordinates.
			// tex3D(obj, tx, ty, tz) where tx/ty/tz are in voxel units (0..vx/vy/vz).
			// The texture sampler performs hardware trilinear interpolation and
			// clamps out-of-bounds accesses to the border value (0).
			//
			// Coordinate mapping (same world-space convention as modes 0/1):
			//   tx = (world_x - xs[0]) / xv   (voxel column index, float)
			//   ty = (world_y - ys[0]) / yv   (voxel row    index, float)
			//   tz = slice index i             (voxel depth  index, float)
			//
			// We sample at the centre of each voxel slab (coordz[i]) and weight
			// by the ray path length through that slab, exactly as in mode 1 but
			// with a single tex3D call replacing 8 global reads + manual lerp.

			for(int i = 0; i < vz; i++)
			{
				// Top and bottom faces of voxel slab i
				float3 newdir  = calcIntersect(dir, source, coordz[i] + zv/2.0f);
				float3 newdir2 = calcIntersect(dir, source, coordz[i] - zv/2.0f);
				l = length(newdir - newdir2);

				if((newdir.x >= xs[0] && newdir.x < xs[1]) && (newdir.y >= ys[0] && newdir.y < ys[1]))
				{
					// Convert world coordinates to voxel-space texture coordinates.
					// +0.5f shifts from voxel-corner to voxel-centre convention used by CUDA textures.
					float tx1 = (newdir.x  - xs[0]) / xv + 0.5f;
					float ty1 = (newdir.y  - ys[0]) / yv + 0.5f;
					float tz1 = (float)i + 0.5f;

					float tx2 = (newdir2.x - xs[0]) / xv + 0.5f;
					float ty2 = (newdir2.y - ys[0]) / yv + 0.5f;
					float tz2 = (float)i + 0.5f;

					// Single hardware trilinear fetch — replaces 8 global reads + manual lerp
					float val1 = tex3D<float>(volTex, tx1, ty1, tz1);
					float val2 = tex3D<float>(volTex, tx2, ty2, tz2);

					if(abs(newdir.x - newdir2.x) <= xv && abs(newdir.y - newdir2.y) <= yv)
					{
						sum += l * ((val1 + val2) / 2.0f);
					}
					else
					{
						float d1 = sqrtf(sqr(newdir.x  - xv*floor(newdir.x /xv)) +
						                 sqr(newdir.y  - yv*floor(newdir.y /yv)));
						sum += d1 * val1 + (l - d1) * val2;
					}
				}
			}
			break;
		}
		default:
			sum = 1.0f;
	}

	result[x + yy*(int)(nu) + z*int(nu*nv)] = sum;
	}
}

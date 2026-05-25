extern "C++" {

inline __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s; a.y *= s; a.z *= s;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

} // extern "C++"

extern "C" {

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

// Compute the intersection of a ray (from source through voxel centre) with the detector plane.
__device__ float3 calcIntersect(float x, float y, float z, float xv, float yv, float zv,
                                float* xs, float* ys, float* zs,
                                float source, float detector, float* TSD, float* TDD)
{
	float3 det;
	float dirx = (x * xv) + xs[0] - source;
	float diry = (y * yv) + ys[0] - TSD[1];
	float dirz = z - TSD[2];

	float len = sqrt(dirx * dirx + diry * diry + dirz * dirz);
	dirx /= len;
	diry /= len;
	dirz /= len;

	float dist = (TDD[2] - TSD[2]) / dirz;

	det.x = source + dirx * dist - detector;
	det.y = TSD[1] + diry * dist;
	det.z = TSD[2] + dirz * dist;
	return det;
}

// Compute the intersection of a normalised ray with a z-plane at depth dz.
__device__ float3 calcSlice(float3 dir, float3 TSD, float dz)
{
	float3 det;
	float dist = (dz - TSD.z) / dir.z;
	det.x = TSD.x + dir.x * dist;
	det.y = TSD.y + dir.y * dist;
	det.z = TSD.z + dir.z * dist;
	return det;
}

// Bilinear interpolation over a 2×2 neighbourhood.
__device__ float bilinear_interpolate(float x, float y, float a, float b, float c, float d)
{
	return a * (1 - x) * (1 - y) + b * x * (1 - y) + c * (1 - x) * y + d * x * y;
}

// Backprojection kernel.
// mode 0 – nearest-neighbour lookup on the detector
// mode 1 – bilinear interpolation on the detector
// mode 2 – hardware trilinear interpolation via 3D texture (sinogram bound as volTex)

__global__ void backprojection(float* result, float* input,
                               float* xs, float* ys, float* zs,
                               float xv, float yv, float zv,
                               int vx, int vy, int vz,
                               float* TSD, float* TDD,
                               float su, float sv,
                               float nu, float nv,
                               int nproj,
                               float* source_coord, float* detector_coord,
                               float* coordz, int mode,
                               cudaTextureObject_t sinoTex)
{
	unsigned short x  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int            y  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	// yy: voxel y-index within a slice; z: slice index
	unsigned short yy = y % vy;
	unsigned short z  = (unsigned short)floor(float(y) / float(vy));

	if (x >= vx || yy >= vy || z >= vz)
		return;

	switch (mode)
	{
		case 0: // Nearest-neighbour
		{
			for (int i = 0; i < nproj; i++)
			{
				float3 dir = make_float3((x * xv) + xs[0] - source_coord[i],
				                        (yy * yv) + ys[0] - TSD[1],
				                        coordz[z] - TSD[2]);
				dir = dir / length(dir);
				float3 n1 = calcSlice(dir, make_float3(source_coord[i], TSD[1], TSD[2]), coordz[z] + zv / 2.0f);
				float3 n2 = calcSlice(dir, make_float3(source_coord[i], TSD[1], TSD[2]), coordz[z] - zv / 2.0f);
				float l = length(n1 - n2);

				float3 det_center = calcIntersect(float(x), float(yy), coordz[z],
				                                  xv, yv, zv, xs, ys, zs,
				                                  source_coord[i], detector_coord[i], TSD, TDD);
				int indx1 = round(det_center.x * (nu / su) + nu / 2);
				int indy1 = round(det_center.y * (nv / sv) + nv / 2);

				if (indx1 >= 0 && indx1 < nu - 1 && indy1 >= 0 && indy1 < nv - 1)
				{
					result[x + yy * vx + z * (vx * vy)] +=
					    l * input[(int)(indx1 + indy1 * (int)nu + i * (int)(nu * nv))];
				}
			}
			break;
		}
		case 1: // Bilinear interpolation
		{
			for (int i = 0; i < nproj; i++)
			{
				float3 dir = make_float3((x * xv) + xs[0] - source_coord[i],
				                        (yy * yv) + ys[0] - TSD[1],
				                        coordz[z] - TSD[2]);
				dir = dir / length(dir);
				float3 n1 = calcSlice(dir, make_float3(source_coord[i], TSD[1], TSD[2]), coordz[z] + zv / 2.0f);
				float3 n2 = calcSlice(dir, make_float3(source_coord[i], TSD[1], TSD[2]), coordz[z] - zv / 2.0f);
				float l = length(n1 - n2);

				float3 det_center = calcIntersect(float(x),        float(yy),        coordz[z], xv, yv, zv, xs, ys, zs, source_coord[i], detector_coord[i], TSD, TDD);
				float3 det_a      = calcIntersect(float(x - xv/2), float(yy - yv/2), coordz[z], xv, yv, zv, xs, ys, zs, source_coord[i], detector_coord[i], TSD, TDD);
				float3 det_b      = calcIntersect(float(x + xv/2), float(yy - yv/2), coordz[z], xv, yv, zv, xs, ys, zs, source_coord[i], detector_coord[i], TSD, TDD);
				float3 det_c      = calcIntersect(float(x - xv/2), float(yy + yv/2), coordz[z], xv, yv, zv, xs, ys, zs, source_coord[i], detector_coord[i], TSD, TDD);
				float3 det_d      = calcIntersect(float(x + xv/2), float(yy + yv/2), coordz[z], xv, yv, zv, xs, ys, zs, source_coord[i], detector_coord[i], TSD, TDD);

				int indxa = round(det_a.x * (nu / su) + nu / 2);
				int indya = round(det_a.y * (nv / sv) + nv / 2);
				int indxb = round(det_b.x * (nu / su) + nu / 2);
				int indyb = round(det_b.y * (nv / sv) + nv / 2);
				int indxc = round(det_c.x * (nu / su) + nu / 2);
				int indyc = round(det_c.y * (nv / sv) + nv / 2);
				int indxd = round(det_d.x * (nu / su) + nu / 2);
				int indyd = round(det_d.y * (nv / sv) + nv / 2);

				if (indxa >= 0 && indxa < nu - 1 && indya >= 0 && indya < nv - 1 &&
				    indxb >= 0 && indxb < nu - 1 && indyb >= 0 && indyb < nv - 1 &&
				    indxc >= 0 && indxc < nu - 1 && indyc >= 0 && indyc < nv - 1 &&
				    indxd >= 0 && indxd < nu - 1 && indyd >= 0 && indyd < nv - 1)
				{
					result[x + yy * vx + z * (vx * vy)] += l * bilinear_interpolate(
					    calc_d(det_center.x), calc_d(det_center.y),
					    input[(int)(indxa + indya * (int)nu + i * (int)(nu * nv))],
					    input[(int)(indxb + indyb * (int)nu + i * (int)(nu * nv))],
					    input[(int)(indxc + indyc * (int)nu + i * (int)(nu * nv))],
					    input[(int)(indxd + indyd * (int)nu + i * (int)(nu * nv))]);
				}
			}
			break;
		}
		case 2: // Hardware trilinear interpolation via 3D texture (sinogram as texture)
		{
			// The sinogram is bound as a 3D texture with shape (nproj, nv, nu).
			// Texture coordinates (non-normalised):
			//   tx = detector u-index  (0 .. nu)
			//   ty = detector v-index  (0 .. nv)
			//   tz = projection index  (0 .. nproj)
			for (int i = 0; i < nproj; i++)
			{
				float3 dir = make_float3((x * xv) + xs[0] - source_coord[i],
				                        (yy * yv) + ys[0] - TSD[1],
				                        coordz[z] - TSD[2]);
				dir = dir / length(dir);
				float3 n1 = calcSlice(dir, make_float3(source_coord[i], TSD[1], TSD[2]), coordz[z] + zv / 2.0f);
				float3 n2 = calcSlice(dir, make_float3(source_coord[i], TSD[1], TSD[2]), coordz[z] - zv / 2.0f);
				float l = length(n1 - n2);

				float3 det_center = calcIntersect(float(x), float(yy), coordz[z],
				                                  xv, yv, zv, xs, ys, zs,
				                                  source_coord[i], detector_coord[i], TSD, TDD);

				// Convert physical detector coordinates to texture coordinates (+0.5 for voxel-centre convention)
				float tx = det_center.x * (nu / su) + nu / 2.0f + 0.5f;
				float ty = det_center.y * (nv / sv) + nv / 2.0f + 0.5f;
				float tz = (float)i + 0.5f;

				float val = tex3D<float>(sinoTex, tx, ty, tz);
				result[x + yy * vx + z * (vx * vy)] += l * val;
			}
			break;
		}
		default:
			result[x + yy * vx + z * (vx * vy)] = 1.0f;
	}
}
}
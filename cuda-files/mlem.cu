// ============================================================================
//  GPU kernels for iterative (MLEM / OSEM) tomosynthesis reconstruction.
//
//  Geometry convention (identical to projection.cu / backprojection.cu):
//    * volume memory layout   : vol[ix + iy*vx + iz*vx*vy]      (C-order (vz, vy, vx))
//    * voxel (ix, iy) centre  : (xs0 + ix*xv, ys0 + iy*yv)
//    * slab iz                : centred at coordz[iz], thickness zv
//    * projection p           : source  S_p = (source[p], TSD.y, TSD.z)
//                               detector pixel (u, v) at
//                               (detector[p] + (u - nu/2)*du, TDD.y + (v - nv/2)*dv, TDD.z)
//    * sinogram memory layout : sino[u + v*nu + p*nu*nv]        (C-order (nproj, nv, nu))
//
//  The ray-driven forward projector samples every slab with `nsub` equidistant
//  points along the ray (nsub adapts to the lateral ray shift inside the slab,
//  i.e. ~one sample per voxel crossing, Joseph-like) and bilinearly
//  interpolates in-plane.  `backproject_ray` is its *exact* adjoint (scatter
//  with atomicAdd).  `backproject_voxel` is a fast, gather-type voxel-driven
//  approximation of the adjoint (no atomics), sampling the voxel's depth extent
//  with the same sub-slab scheme.
//
//  Every kernel receives a projection *list* (proj_ids, nproj_sub) so that the
//  same code serves full MLEM and ordered-subset EM: the sinogram buffers are
//  always the full (nproj, nv, nu) arrays, only the listed projections are
//  touched.
// ============================================================================

#define MAX_NSUB 32

struct Geom {
    float xs0, ys0;          // lower bounds of the volume (x, y)
    float xv, yv, zv;        // voxel sizes
    int   vx, vy, vz;        // voxel counts
    float tsd_y, tsd_z;      // source y / z
    float tdd_y, tdd_z;      // detector y / z
    float du, dv;            // detector pixel pitch
    int   nu, nv;            // detector pixel counts
};

__constant__ Geom   g;
__constant__ float  c_coordz[2048];
__constant__ float  c_source[1024];
__constant__ float  c_detector[1024];

__device__ __forceinline__ int adaptive_nsub(float dxdz, float dydz)
{
    // lateral shift of the ray inside one slab, in voxel units
    float shift = fmaxf(fabsf(dxdz) * g.zv / g.xv, fabsf(dydz) * g.zv / g.yv);
    int n = (int)ceilf(shift) + 1;
    return min(max(n, 1), MAX_NSUB);
}

// ---------------------------------------------------------------------------
// Forward projection  q = A x   (ray driven)
// grid: x -> u, y -> v, z -> index into proj_ids
// ---------------------------------------------------------------------------
extern "C" __global__ void forward_project(float* __restrict__ sino,
                                           const float* __restrict__ vol,
                                           const int* __restrict__ proj_ids)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int p = proj_ids[blockIdx.z];
    if (u >= g.nu || v >= g.nv) return;

    float sx = c_source[p], sy = g.tsd_y, sz = g.tsd_z;
    float px = c_detector[p] + ((float)u - g.nu * 0.5f) * g.du;
    float py = g.tdd_y + ((float)v - g.nv * 0.5f) * g.dv;
    float pz = g.tdd_z;

    float dx = px - sx, dy = py - sy, dz = pz - sz;
    float len = sqrtf(dx * dx + dy * dy + dz * dz);
    float dxdz = dx / dz, dydz = dy / dz;
    float seg = g.zv * len / fabsf(dz);           // path length through one slab
    int   nsub = adaptive_nsub(dxdz, dydz);
    float w = seg / (float)nsub;
    float inv_xv = 1.0f / g.xv, inv_yv = 1.0f / g.yv;

    float sum = 0.0f;
    for (int iz = 0; iz < g.vz; ++iz) {
        const float* slab = vol + (size_t)iz * g.vx * g.vy;
        float z0 = c_coordz[iz] - 0.5f * g.zv;
        for (int k = 0; k < nsub; ++k) {
            float z = z0 + ((float)k + 0.5f) * g.zv / (float)nsub;
            float t = z - sz;
            float fx = (sx + dxdz * t - g.xs0) * inv_xv;   // continuous voxel index
            float fy = (sy + dydz * t - g.ys0) * inv_yv;
            int ix = (int)floorf(fx), iy = (int)floorf(fy);
            if (ix < -1 || iy < -1 || ix >= g.vx || iy >= g.vy) continue;
            float ax = fx - ix, ay = fy - iy;
            float v00 = (ix >= 0 && iy >= 0)               ? slab[ix + iy * g.vx] : 0.f;
            float v10 = (ix + 1 < g.vx && iy >= 0)         ? slab[ix + 1 + iy * g.vx] : 0.f;
            float v01 = (ix >= 0 && iy + 1 < g.vy)         ? slab[ix + (iy + 1) * g.vx] : 0.f;
            float v11 = (ix + 1 < g.vx && iy + 1 < g.vy)   ? slab[ix + 1 + (iy + 1) * g.vx] : 0.f;
            sum += w * ((1.f - ax) * ((1.f - ay) * v00 + ay * v01) +
                        ax         * ((1.f - ay) * v10 + ay * v11));
        }
    }
    sino[(size_t)u + (size_t)v * g.nu + (size_t)p * g.nu * g.nv] = sum;
}

// ---------------------------------------------------------------------------
// Exact adjoint  x += A^T r  (ray driven scatter, atomics)
// ---------------------------------------------------------------------------
extern "C" __global__ void backproject_ray(float* __restrict__ vol,
                                           const float* __restrict__ sino,
                                           const int* __restrict__ proj_ids)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int p = proj_ids[blockIdx.z];
    if (u >= g.nu || v >= g.nv) return;

    float val = sino[(size_t)u + (size_t)v * g.nu + (size_t)p * g.nu * g.nv];
    if (val == 0.0f) return;

    float sx = c_source[p], sy = g.tsd_y, sz = g.tsd_z;
    float px = c_detector[p] + ((float)u - g.nu * 0.5f) * g.du;
    float py = g.tdd_y + ((float)v - g.nv * 0.5f) * g.dv;
    float pz = g.tdd_z;

    float dx = px - sx, dy = py - sy, dz = pz - sz;
    float len = sqrtf(dx * dx + dy * dy + dz * dz);
    float dxdz = dx / dz, dydz = dy / dz;
    float seg = g.zv * len / fabsf(dz);
    int   nsub = adaptive_nsub(dxdz, dydz);
    float wv = val * seg / (float)nsub;
    float inv_xv = 1.0f / g.xv, inv_yv = 1.0f / g.yv;

    for (int iz = 0; iz < g.vz; ++iz) {
        float* slab = vol + (size_t)iz * g.vx * g.vy;
        float z0 = c_coordz[iz] - 0.5f * g.zv;
        for (int k = 0; k < nsub; ++k) {
            float z = z0 + ((float)k + 0.5f) * g.zv / (float)nsub;
            float t = z - sz;
            float fx = (sx + dxdz * t - g.xs0) * inv_xv;
            float fy = (sy + dydz * t - g.ys0) * inv_yv;
            int ix = (int)floorf(fx), iy = (int)floorf(fy);
            if (ix < -1 || iy < -1 || ix >= g.vx || iy >= g.vy) continue;
            float ax = fx - ix, ay = fy - iy;
            if (ix >= 0 && iy >= 0)             atomicAdd(&slab[ix + iy * g.vx],           wv * (1.f - ax) * (1.f - ay));
            if (ix + 1 < g.vx && iy >= 0)       atomicAdd(&slab[ix + 1 + iy * g.vx],       wv * ax * (1.f - ay));
            if (ix >= 0 && iy + 1 < g.vy)       atomicAdd(&slab[ix + (iy + 1) * g.vx],     wv * (1.f - ax) * ay);
            if (ix + 1 < g.vx && iy + 1 < g.vy) atomicAdd(&slab[ix + 1 + (iy + 1) * g.vx], wv * ax * ay);
        }
    }
}

// ---------------------------------------------------------------------------
// Approximate adjoint  x += B r  (voxel driven gather, no atomics)
//
// For voxel j and projection p the ray-driven adjoint sums, over all rays that
// pass near the voxel, (seg/nsub) * bilinear_weight.  The density of rays per
// unit voxel area at depth z is  m(z)^2 / (du*dv)  with the magnification
// m(z) = (S.z - D.z)/(S.z - z), hence each depth sample is weighted with
//      (seg/nsub) * xv*yv * m(z)^2 / (du*dv)
// which makes B ~= A^T (verified by the adjoint test in the Python wrapper).
// grid: x -> ix, y -> iy, z -> iz
// ---------------------------------------------------------------------------
extern "C" __global__ void backproject_voxel(float* __restrict__ vol,
                                             const float* __restrict__ sino,
                                             const int* __restrict__ proj_ids,
                                             int nproj_sub)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (ix >= g.vx || iy >= g.vy) return;

    float wx = g.xs0 + ix * g.xv;
    float wy = g.ys0 + iy * g.yv;
    float zc = c_coordz[iz];
    float sz = g.tsd_z, sy = g.tsd_y;
    float H  = g.tdd_z - sz;                       // signed source->detector distance
    float area = g.xv * g.yv / (g.du * g.dv);
    float inv_du = 1.0f / g.du, inv_dv = 1.0f / g.dv;

    float acc = 0.0f;
    for (int q = 0; q < nproj_sub; ++q) {
        int p = proj_ids[q];
        float sx = c_source[p];
        const float* img = sino + (size_t)p * g.nu * g.nv;

        // direction through the voxel centre -> slab path length / nsub
        float dx = wx - sx, dy = wy - sy, dz = zc - sz;
        float len = sqrtf(dx * dx + dy * dy + dz * dz);
        float seg = g.zv * len / fabsf(dz);
        int nsub = adaptive_nsub(dx / dz, dy / dz);

        for (int k = 0; k < nsub; ++k) {
            float z = zc - 0.5f * g.zv + ((float)k + 0.5f) * g.zv / (float)nsub;
            float s = H / (z - sz);                // magnification m(z)
            float detx = sx + (wx - sx) * s - c_detector[p];
            float dety = sy + (wy - sy) * s - g.tdd_y;
            float fu = detx * inv_du + g.nu * 0.5f;
            float fv = dety * inv_dv + g.nv * 0.5f;
            int iu = (int)floorf(fu), iv = (int)floorf(fv);
            if (iu < -1 || iv < -1 || iu >= g.nu || iv >= g.nv) continue;
            float au = fu - iu, av = fv - iv;
            float p00 = (iu >= 0 && iv >= 0)               ? img[iu + iv * g.nu] : 0.f;
            float p10 = (iu + 1 < g.nu && iv >= 0)         ? img[iu + 1 + iv * g.nu] : 0.f;
            float p01 = (iu >= 0 && iv + 1 < g.nv)         ? img[iu + (iv + 1) * g.nu] : 0.f;
            float p11 = (iu + 1 < g.nu && iv + 1 < g.nv)   ? img[iu + 1 + (iv + 1) * g.nu] : 0.f;
            float val = (1.f - au) * ((1.f - av) * p00 + av * p01) +
                        au         * ((1.f - av) * p10 + av * p11);
            acc += val * (seg / (float)nsub) * area * s * s;
        }
    }
    vol[(size_t)ix + (size_t)iy * g.vx + (size_t)iz * g.vx * g.vy] += acc;
}

// ---------------------------------------------------------------------------
// Element-wise helpers (operate on a contiguous range of `n` elements)
// ---------------------------------------------------------------------------

// r = w * y / max(q, eps)     (0 where w == 0)
extern "C" __global__ void ratio_kernel(float* __restrict__ r,
                                        const float* __restrict__ y,
                                        const float* __restrict__ q,
                                        const float* __restrict__ w,
                                        float eps, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float wi = w[i];
    r[i] = (wi > 0.f) ? wi * y[i] / fmaxf(q[i], eps) : 0.f;
}

// x = mask ? x * bp / sens : x      (sens <= sens_min -> frozen)
extern "C" __global__ void update_kernel(float* __restrict__ x,
                                         const float* __restrict__ bp,
                                         const float* __restrict__ sens,
                                         float sens_min, float max_step, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = sens[i];
    if (s > sens_min) {
        float f = bp[i] / s;
        if (max_step > 0.f) f = fminf(fmaxf(f, 1.0f / max_step), max_step);
        x[i] *= f;
    }
}

// ============================================================================
//  Separable-footprint / distance-driven projector pair
//  (after Long, Fessler & Balter, IEEE TMI 2010 and its GPU implementation by
//   Meng & Fessler, "GPU acceleration of 3D forward and backward projection
//   using separable footprints").
//
//  In tomosynthesis every slab is parallel to the detector, so on a plane at
//  depth z the perspective image of a voxel and of a detector cell are both
//  axis-aligned rectangles: the footprint is *exactly* separable
//  (rect x rect) on each plane.  The voxel depth extent (trapezoid in u for
//  oblique rays) is integrated with c_nsub[p] sub-planes per slab.
//
//  Weight of voxel j for detector cell i, sub-plane k of slab iz:
//      a_ij = (zv/nsub) * sec(theta_i) * (ox/du) * (oy/dv)
//  where ox, oy are the overlaps of the voxel and the detector cell measured
//  in detector coordinates (equivalently overlap on the plane divided by the
//  cell footprint on that plane).  Both kernels evaluate the same a_ij, the
//  forward one as a gather over voxels, the back one as a gather over
//  detector cells: an exactly matched pair without atomics, modelling the
//  finite voxel *and* detector cell size (no point sampling / moire).
// ============================================================================

__constant__ int c_nsub[1024];

#define FP_MAX_SPAN 64

extern "C" __global__ void forward_footprint(float* __restrict__ sino,
                                             const float* __restrict__ vol,
                                             const int* __restrict__ proj_ids)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int p = proj_ids[blockIdx.z];
    if (u >= g.nu || v >= g.nv) return;

    float sx = c_source[p], sy = g.tsd_y, sz = g.tsd_z;
    float px = c_detector[p] + ((float)u - g.nu * 0.5f) * g.du;
    float py = g.tdd_y + ((float)v - g.nv * 0.5f) * g.dv;
    float H  = g.tdd_z - sz;
    float sec = sqrtf((px - sx) * (px - sx) + (py - sy) * (py - sy) + H * H) / fabsf(H);
    int   nsub = c_nsub[p];
    float dzs = g.zv / (float)nsub;
    float wz = dzs * sec;
    float xlo = g.xs0 - 0.5f * g.xv, ylo = g.ys0 - 0.5f * g.yv;
    float inv_xv = 1.0f / g.xv, inv_yv = 1.0f / g.yv;
    float ax0 = px - 0.5f * g.du - sx, ax1 = px + 0.5f * g.du - sx;
    float ay0 = py - 0.5f * g.dv - sy, ay1 = py + 0.5f * g.dv - sy;

    float sum = 0.0f;
    for (int iz = 0; iz < g.vz; ++iz) {
        const float* slab = vol + (size_t)iz * g.vx * g.vy;
        float z0 = c_coordz[iz] - 0.5f * g.zv;
        for (int k = 0; k < nsub; ++k) {
            float t = (z0 + ((float)k + 0.5f) * dzs - sz) / H;   // > 0
            float fx0 = sx + ax0 * t, fx1 = sx + ax1 * t;        // cell footprint on the plane
            float fy0 = sy + ay0 * t, fy1 = sy + ay1 * t;
            int ix0 = max((int)floorf((fx0 - xlo) * inv_xv), 0);
            int ix1 = min((int)floorf((fx1 - xlo) * inv_xv), g.vx - 1);
            int iy0 = max((int)floorf((fy0 - ylo) * inv_yv), 0);
            int iy1 = min((int)floorf((fy1 - ylo) * inv_yv), g.vy - 1);
            if (ix0 > ix1 || iy0 > iy1) continue;
            ix1 = min(ix1, ix0 + FP_MAX_SPAN);
            iy1 = min(iy1, iy0 + FP_MAX_SPAN);
            float acc = 0.0f;
            for (int iy = iy0; iy <= iy1; ++iy) {
                float e0 = ylo + iy * g.yv;
                float oy = fminf(fy1, e0 + g.yv) - fmaxf(fy0, e0);
                const float* row = slab + (size_t)iy * g.vx;
                float racc = 0.0f;
                for (int ix = ix0; ix <= ix1; ++ix) {
                    float x0 = xlo + ix * g.xv;
                    racc += (fminf(fx1, x0 + g.xv) - fmaxf(fx0, x0)) * row[ix];
                }
                acc += oy * racc;
            }
            sum += wz * acc / ((fx1 - fx0) * (fy1 - fy0));
        }
    }
    sino[(size_t)u + (size_t)v * g.nu + (size_t)p * g.nu * g.nv] = sum;
}

extern "C" __global__ void backproject_footprint(float* __restrict__ vol,
                                                 const float* __restrict__ sino,
                                                 const int* __restrict__ proj_ids,
                                                 int nproj_sub)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (ix >= g.vx || iy >= g.vy) return;

    float xe0 = g.xs0 + ((float)ix - 0.5f) * g.xv, xe1 = xe0 + g.xv;
    float ye0 = g.ys0 + ((float)iy - 0.5f) * g.yv, ye1 = ye0 + g.yv;
    float sy = g.tsd_y, sz = g.tsd_z;
    float H = g.tdd_z - sz;
    float H2 = H * H, invH = 1.0f / fabsf(H);
    float inv_du = 1.0f / g.du, inv_dv = 1.0f / g.dv;
    float z0 = c_coordz[iz] - 0.5f * g.zv;

    float acc = 0.0f;
    for (int q = 0; q < nproj_sub; ++q) {
        int p = proj_ids[q];
        float sx = c_source[p], detx = c_detector[p];
        const float* img = sino + (size_t)p * g.nu * g.nv;
        int nsub = c_nsub[p];
        float dzs = g.zv / (float)nsub;
        for (int k = 0; k < nsub; ++k) {
            float t = (z0 + ((float)k + 0.5f) * dzs - sz) / H;
            float it = 1.0f / t;
            // voxel edges projected to detector coordinates (cell-grid frame)
            float X0 = sx + (xe0 - sx) * it - detx, X1 = sx + (xe1 - sx) * it - detx;
            float Y0 = sy + (ye0 - sy) * it - g.tdd_y, Y1 = sy + (ye1 - sy) * it - g.tdd_y;
            int u0 = max((int)floorf(X0 * inv_du + g.nu * 0.5f + 0.5f), 0);
            int u1 = min((int)floorf(X1 * inv_du + g.nu * 0.5f + 0.5f), g.nu - 1);
            int v0 = max((int)floorf(Y0 * inv_dv + g.nv * 0.5f + 0.5f), 0);
            int v1 = min((int)floorf(Y1 * inv_dv + g.nv * 0.5f + 0.5f), g.nv - 1);
            if (u0 > u1 || v0 > v1) continue;
            u1 = min(u1, u0 + FP_MAX_SPAN);
            v1 = min(v1, v0 + FP_MAX_SPAN);
            float kacc = 0.0f;
            for (int vv = v0; vv <= v1; ++vv) {
                float c0 = ((float)vv - g.nv * 0.5f - 0.5f) * g.dv;
                float oy = fminf(Y1, c0 + g.dv) - fmaxf(Y0, c0);
                float dyc = g.tdd_y + ((float)vv - g.nv * 0.5f) * g.dv - sy;
                const float* row = img + (size_t)vv * g.nu;
                for (int uu = u0; uu <= u1; ++uu) {
                    float c1 = ((float)uu - g.nu * 0.5f - 0.5f) * g.du;
                    float ox = fminf(X1, c1 + g.du) - fmaxf(X0, c1);
                    float dxc = detx + ((float)uu - g.nu * 0.5f) * g.du - sx;
                    float sec = sqrtf(dxc * dxc + dyc * dyc + H2) * invH;
                    kacc += row[uu] * sec * ox * oy;
                }
            }
            acc += kacc * dzs * inv_du * inv_dv;
        }
    }
    vol[(size_t)ix + (size_t)iy * g.vx + (size_t)iz * g.vx * g.vy] += acc;
}

// ============================================================================
//  Edge-preserving regulariser for OS-LALM (PWLS)
//     R(x) = sum_j sum_{k in N+(j)} beta_k * psi(x_j - x_k),
//     psi(t) = delta^2 (sqrt(1 + (t/delta)^2) - 1)      (hyperbola / Fair)
//  Gradient and the De Pierro separable quadratic surrogate curvature
//     d_j = sum_k 2 beta_k omega(x_j - x_k),  omega(t) = psi'(t)/t <= 1
//  over the 6-neighbourhood (beta_xy in-plane, beta_z between slabs).
// ============================================================================
__device__ __forceinline__ void reg_term(float d, float beta, float inv_d2, float& gr, float& cv)
{
    float w = rsqrtf(1.0f + d * d * inv_d2);    // omega(t) = psi'(t)/t
    gr += beta * w * d;
    cv += 2.0f * beta * w;
}

extern "C" __global__ void reg_grad_curv(float* __restrict__ grad, float* __restrict__ curv,
                                         const float* __restrict__ x,
                                         float beta_xy, float beta_z, float delta)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (ix >= g.vx || iy >= g.vy) return;
    size_t sxy = (size_t)g.vx * g.vy;
    size_t j = (size_t)ix + (size_t)iy * g.vx + (size_t)iz * sxy;
    float xj = x[j], gr = 0.f, cv = 0.f, id2 = 1.0f / (delta * delta);
    if (ix > 0)        reg_term(xj - x[j - 1],    beta_xy, id2, gr, cv);
    if (ix < g.vx - 1) reg_term(xj - x[j + 1],    beta_xy, id2, gr, cv);
    if (iy > 0)        reg_term(xj - x[j - g.vx], beta_xy, id2, gr, cv);
    if (iy < g.vy - 1) reg_term(xj - x[j + g.vx], beta_xy, id2, gr, cv);
    if (beta_z > 0.f) {
        if (iz > 0)        reg_term(xj - x[j - sxy], beta_z, id2, gr, cv);
        if (iz < g.vz - 1) reg_term(xj - x[j + sxy], beta_z, id2, gr, cv);
    }
    grad[j] = gr;
    curv[j] = cv;
}

// ============================================================================
//  Total variation (TV) kernels
//
//  TV(x) = sum_j sqrt( (D_x x)_j^2 + (D_y x)_j^2 + (w_z D_z x)_j^2 )
//  forward differences, Neumann boundary; in-plane isotropic, the depth
//  component is weighted with w_z (0 = slice-wise 2D TV).
//
//  * tv_grad          : gradient of the eps-smoothed TV (Sidky, Kao & Pan,
//                       arXiv:0904.4495, used for the TV steepest-descent
//                       steps of the OSEM-TV variant)
//  * tv_dual_update / tv_primal_update : primal-dual (Chambolle-Pock) solver
//                       of the weighted TV proximal problem
//                         min_{x>=0} 1/2 ||x - z||_W^2 + lambda TV(x)
//                       used as the inner step of OS-LALM-TV.  The dual
//                       projection onto ||p|| <= lambda is the dual form of
//                       the shrinkage step of Xiao & Yang (FTVCS) / the TV
//                       denoising step of Jia et al.
// ============================================================================

__device__ __forceinline__ float3 fwd_diff(const float* __restrict__ x, size_t j,
                                           int ix, int iy, int iz, float wz)
{
    size_t sxy = (size_t)g.vx * g.vy;
    float xj = x[j];
    float3 d;
    d.x = (ix < g.vx - 1) ? x[j + 1] - xj : 0.f;
    d.y = (iy < g.vy - 1) ? x[j + g.vx] - xj : 0.f;
    d.z = (wz > 0.f && iz < g.vz - 1) ? wz * (x[j + sxy] - xj) : 0.f;
    return d;
}

extern "C" __global__ void tv_grad(float* __restrict__ grad, const float* __restrict__ x,
                                   float wz, float eps)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (ix >= g.vx || iy >= g.vy) return;
    size_t sxy = (size_t)g.vx * g.vy;
    size_t j = (size_t)ix + (size_t)iy * g.vx + (size_t)iz * sxy;
    // d/dx_j of sum_k sqrt(eps + |D x|_k^2): own term + the terms of the
    // backward neighbours that use x_j as their forward sample
    float3 d = fwd_diff(x, j, ix, iy, iz, wz);
    float gr = -(d.x + d.y + wz * d.z) * rsqrtf(eps + d.x * d.x + d.y * d.y + d.z * d.z);
    if (ix > 0) {
        float3 e = fwd_diff(x, j - 1, ix - 1, iy, iz, wz);
        gr += e.x * rsqrtf(eps + e.x * e.x + e.y * e.y + e.z * e.z);
    }
    if (iy > 0) {
        float3 e = fwd_diff(x, j - g.vx, ix, iy - 1, iz, wz);
        gr += e.y * rsqrtf(eps + e.x * e.x + e.y * e.y + e.z * e.z);
    }
    if (wz > 0.f && iz > 0) {
        float3 e = fwd_diff(x, j - sxy, ix, iy, iz - 1, wz);
        gr += wz * e.z * rsqrtf(eps + e.x * e.x + e.y * e.y + e.z * e.z);
    }
    grad[j] = gr;
}

// p <- proj_{|p| <= lambda} (p + sigma * D xbar)
extern "C" __global__ void tv_dual_update(float* __restrict__ px, float* __restrict__ py,
                                          float* __restrict__ pz, const float* __restrict__ xbar,
                                          float sigma, float lambda, float wz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (ix >= g.vx || iy >= g.vy) return;
    size_t j = (size_t)ix + (size_t)iy * g.vx + (size_t)iz * g.vx * g.vy;
    float3 d = fwd_diff(xbar, j, ix, iy, iz, wz);
    float a = px[j] + sigma * d.x, b = py[j] + sigma * d.y, c = pz[j] + sigma * d.z;
    float s = fmaxf(1.0f, sqrtf(a * a + b * b + c * c) / lambda);
    px[j] = a / s; py[j] = b / s; pz[j] = c / s;
}

// x <- max(0, (tau*w*z + x + tau*div p) / (tau*w + 1)),  xbar <- 2 x_new - x_old
extern "C" __global__ void tv_primal_update(float* __restrict__ x, float* __restrict__ xbar,
                                            const float* __restrict__ px, const float* __restrict__ py,
                                            const float* __restrict__ pz, const float* __restrict__ z,
                                            const float* __restrict__ w, const float* __restrict__ m,
                                            float tau, float wz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (ix >= g.vx || iy >= g.vy) return;
    size_t sxy = (size_t)g.vx * g.vy;
    size_t j = (size_t)ix + (size_t)iy * g.vx + (size_t)iz * sxy;
    // div p = -D^T p  (adjoint of the forward differences)
    float div = ((ix < g.vx - 1) ? px[j] : 0.f) - ((ix > 0) ? px[j - 1] : 0.f)
              + ((iy < g.vy - 1) ? py[j] : 0.f) - ((iy > 0) ? py[j - g.vx] : 0.f);
    if (wz > 0.f)
        div += wz * (((iz < g.vz - 1) ? pz[j] : 0.f) - ((iz > 0) ? pz[j - sxy] : 0.f));
    float xo = x[j];
    float xn = (m[j] > 0.f) ? fmaxf(0.f, (tau * w[j] * z[j] + xo + tau * div) / (tau * w[j] + 1.f)) : 0.f;
    x[j] = xn;
    xbar[j] = 2.f * xn - xo;
}

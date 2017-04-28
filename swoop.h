#pragma once

#include <visionaray/math/math.h>

#include <visionaray/tags.h>
#include <visionaray/get_normal.h>
#include <visionaray/get_shading_normal.h>
#include <visionaray/get_tex_coord.h>
#include <visionaray/bvh.h> // FIXME needs to be included before intersector.h
#include <visionaray/intersector.h>

using namespace visionaray;

namespace QUAD_NS
{

template <typename T>
class swoop_quad //: public visionaray::primitive<unsigned>
{
public:

    using scalar_type =  T;
    using vec_type    =  vector<3, T>;
    using matrix_type =  std::array<vec_type, 3>;

public:

    // vertex ordering:
    //
    //    v3
    //       x
    //       | \
    //       |   \
    //       |     \
    //       |       \
    //       |         \
    //       |           x  v4
    //       |           |
    //       |           |
    //       |           |
    //       |           |
    //       x-----------x
    //    v1                v2
    //
    MATH_FUNC swoop_quad() = default;
    MATH_FUNC swoop_quad(
            vector<3, T> const& v1,
            vector<3, T> const& v2,
            vector<3, T> const& v3,
            vector<3, T> const& v4
            )
    {
        // if this quad should be degenerated to a triangle, make sure that v3 == v4
        // v2 == v4 would also be possible, but would need additional checks
        // and a rewrite of the following code

        // calculate transformation data for the triangle (v1, v2, v3)

        vec_type n = normalize(cross(v2-v1, v3-v1));

        // normal = n;

        vec_type c0 = v2 - v1;
        vec_type c1 = v3 - v1;
        vec_type c2 =  n - v1;

        matrix_type inv_transformation = {
                vec_type( c0[0], c1[0], c2[0]),
                vec_type( c0[1], c1[1], c2[1]),
                vec_type( c0[2], c1[2], c2[2])
            };

        // scalar_type det = c0[0] * c1[1] * c2[2] + c1[0] * c2[1] * c0[2] + c2[0] * c0[1] * c1[2]
        //                 - c0[1] * c1[0] * c2[2] - c0[0] * c2[1] * c1[2] - c0[2] * c1[1] * c2[0];
        //
        // // matrix_type inv_transformation_transposed = {
        // //         vec_type( c0[0], c0[1], c0[2]),
        // //         vec_type( c1[0], c1[1], c1[2]),
        // //         vec_type( c2[0], c2[1], c2[2])
        // //     };
        //
        // // non-inverted matrix
        // t_mat = {
        //         vec_type( (c1[1]*c2[2] - c1[2]*c2[1])/det,-(c1[0]*c2[2] - c1[2]*c2[0])/det, (c1[0]*c2[1] - c1[1]*c2[0])/det ),
        //         vec_type(-(c0[1]*c2[2] - c0[2]*c2[1])/det, (c0[0]*c2[2] - c0[2]*c2[0])/det,-(c0[0]*c2[1] - c0[1]*c2[0])/det ),
        //         vec_type( (c0[1]*c1[2] - c0[2]*c1[1])/det,-(c0[0]*c1[2] - c0[2]*c1[0])/det, (c0[0]*c1[1] - c0[1]*c1[0])/det )
        //     };
        //
        // // non-inverted translation
        // t_vec = vec_type(
        //         -dot(t_mat[0], v1),
        //         -dot(t_mat[1], v1),
        //         -dot(t_mat[2], v1)
        //     );

        auto r = this->get_inv_transformation(inv_transformation, v1);

        t_mat = r.first;
        t_vec = r.second;

        // calculate (u,v)-coordinates of the 4th vector relative to the triangle (v1, v2, v3)

        // edge case, degenerated quad
        if (v3 == v4)
        {
            // ensure this quad is exactly a triangle
            this->v4= vector<2, T>(0., 1.);
        }
        else
        {
            this->v4 = vector<2, T>(
                    dot(t_mat[0], v4) + t_vec[0],
                    dot(t_mat[1], v4) + t_vec[1]
                );
        }
    }

    matrix_type t_mat;
    vec_type t_vec;
    vector<2, scalar_type> v4;

    // vec_type normal;

    static std::pair<matrix_type, vec_type> get_inv_transformation(matrix_type mat, vec_type v)
    {
        // matrix_type mat = {
        //         vec_type( mat[0][0], mat[0][1], mat[0][2]),
        //         vec_type( mat[1][0], mat[1][1], mat[1][2]),
        //         vec_type( mat[2][0], mat[2][1], mat[2][2])
        //     };

        scalar_type det = mat[0][0] * mat[1][1] * mat[2][2] + mat[0][1] * mat[1][2] * mat[2][0] + mat[0][2] * mat[1][0] * mat[2][1]
                        - mat[1][0] * mat[0][1] * mat[2][2] - mat[0][0] * mat[1][2] * mat[2][1] - mat[2][0] * mat[1][1] * mat[0][2];

        // non-inverted matrix
        matrix_type inv_mat = std::array<vec_type, 3>{
                vec_type( (mat[1][1]*mat[2][2] - mat[2][1]*mat[1][2])/det,-(mat[0][1]*mat[2][2] - mat[2][1]*mat[0][2])/det, (mat[0][1]*mat[1][2] - mat[1][1]*mat[0][2])/det ),
                vec_type(-(mat[1][0]*mat[2][2] - mat[2][0]*mat[1][2])/det, (mat[0][0]*mat[2][2] - mat[2][0]*mat[0][2])/det,-(mat[0][0]*mat[1][2] - mat[1][0]*mat[0][2])/det ),
                vec_type( (mat[1][0]*mat[2][1] - mat[2][0]*mat[1][1])/det,-(mat[0][0]*mat[2][1] - mat[2][0]*mat[0][1])/det, (mat[0][0]*mat[1][1] - mat[1][0]*mat[0][1])/det )
            };

        // non-inverted translation
        vec_type inv_vec = vec_type(
                -dot(inv_mat[0], v),
                -dot(inv_mat[1], v),
                -dot(inv_mat[2], v)
            );

        return std::make_pair(inv_mat, inv_vec);
    }

    vec_type inv_trans(vec_type v) const
    {
        auto r = this->get_inv_transformation(t_mat, t_vec);

        matrix_type inv_mat = r.first;
        vec_type inv_vec = r.second;

        return vec_type(
                dot(inv_mat[0], v),
                dot(inv_mat[1], v),
                dot(inv_mat[2], v)
            ) + inv_vec;
    }

    vec_type get_v1() const { return inv_trans(vec_type(T(0.0), T(0.0), T(0.0))); }
    vec_type get_v2() const { return inv_trans(vec_type(T(1.0), T(0.0), T(0.0))); }
    vec_type get_v3() const { return inv_trans(vec_type(T(0.0), T(1.0), T(0.0))); }
    vec_type get_v4() const { return inv_trans(vec_type(T(1.0), T(1.0), T(0.0))); }


    // need to reorder vertices for degenerated quads
    //
    // vertex ordering:
    //
    //    v4
    //       x
    //       | \
    //       |   \
    //       |     \
    //       |       \
    //       |         \
    //       |           x  v3
    //       |           |
    //       |           |
    //       |           |
    //       |           |
    //       x-----------x
    //    v1                v2
    //
    static swoop_quad<float> make_quad(
                vector<3, float> const& v1,
                vector<3, float> const& v2,
                vector<3, float> const& v3,
                vector<3, float> const& v4,
                float epsilon=0.01f
            )
    {
        // ensure degenerated quads are exactly a triangle and the matching
        // vertices are v3 and v4
        if (norm(v1-v2) < epsilon)
            return swoop_quad<float>(v3, v4, v2, v2);
        else if (norm(v2-v3) < epsilon)
            return swoop_quad<float>(v4, v1, v3, v3);
        else if (norm(v3-v4) < epsilon)
            return swoop_quad<float>(v1, v2, v4, v4);
        else if (norm(v4-v1) < epsilon)
            return swoop_quad<float>(v2, v3, v1, v1);

        else
            return swoop_quad<float>(v1, v2, v4, v3);
    }
};

namespace detail
{

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect_swoop(
        basic_ray<T> const&                     ray,
        QUAD_NS::swoop_quad<U> const&               quad
        )
{
    vector<2, T> v4(quad.v4);

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.t = T(-1.0);

    vector<3, T> trans_ori = vector<3, T>(
            dot(vector<3, T>(quad.t_mat[0]), ray.ori),
            dot(vector<3, T>(quad.t_mat[1]), ray.ori),
            dot(vector<3, T>(quad.t_mat[2]), ray.ori)
        ) + vector<3, T>(quad.t_vec);

    vector<3, T> trans_dir = vector<3, T>(
            dot(vector<3, T>(quad.t_mat[0]), ray.dir),
            dot(vector<3, T>(quad.t_mat[1]), ray.dir),
            dot(vector<3, T>(quad.t_mat[2]), ray.dir)
        );

    T t = -(trans_ori.z / trans_dir.z);
    T b1 = trans_ori.x + t * trans_dir.x;
    T b2 = trans_ori.y + t * trans_dir.y;

    // (b1,b2) are (u,v)-coordinates (relative to the triangle (v1,v2,v3)) of
    // the intersection point - check if they lie inside quad

    result.hit = b2 >= T(0.0) && b1 >= T(0.0);

    if ( !any(result.hit) )
    {
        return result;
    }

    result.hit &= ( b2 <= ((v4.y-T(1.0)) / v4.x) * b1 + T(1.0) ) || v4.x == T(0.0);
    result.hit &= ( b1 <= ((v4.x-T(1.0)) / v4.y) * b2 + T(1.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

#if CALCULATE_UV
    // now calculate bilinear coordinates relative to the quad
    T u, v;

    // special cases:
    if (quad.v4.x == T(1.0))
    {
        u = b1;
        v = b2 / (u * (v4.y - T(1.0)) + T(1.0));
    }
    else if (quad.v4.y == T(1.0))
    {
        v = b2;
        u = b1 / (v * (v4.x - T(1.0)) + T(1.0));
    }
    else
    {
        // solve A*u^2 + B*u + C = 0
        T A = -(v4.y - T(1.0));
        T B = b1 * (v4.y - T(1.0)) - b2 * (v4.x - T(1.0)) - T(1.0);
        T C = b1;

        T D = B * B - T(4.0) * A * C;
        T Q = -T(0.5) * (B + copysign(sqrt(D), B));

        u = Q / A;
        u = select(u < T(0.0) || u > T(1.0), C / Q, u);
        v = b2 / (u * (v4.y - T(1.0)) + T(1.0));
    }

    result.u = u;
    result.v = v;
#endif

    // result.prim_id = quad.prim_id;
    // result.geom_id = quad.geom_id;
    result.t = t;
    return result;

}

} // detail

template <typename T, typename U>
MATH_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect(
        basic_ray<T> const&                     ray,
        QUAD_NS::swoop_quad<U> const&               quad
        )
{
    return detail::intersect_swoop(ray, quad);
}

struct quad_intersector_swoop : basic_intersector<quad_intersector_swoop>
{
    using basic_intersector<quad_intersector_swoop>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            swoop_quad<S> const& quad
            )
        -> decltype( detail::intersect_swoop(ray, quad) )
    {
        return detail::intersect_swoop(ray, quad);
    }
};

} // QUAD_NS


namespace visionaray
{


template <typename T>
MATH_FUNC
basic_aabb<T> get_bounds(QUAD_NS::swoop_quad<T> const& t)
{
    basic_aabb<T> bounds;

    auto v1 = t.get_v1();
    auto v2 = t.get_v2();
    auto v3 = t.get_v3();
    auto v4 = t.get_v4();

    bounds.invalidate();
    bounds.insert(v1);
    bounds.insert(v2);
    bounds.insert(v3);
    bounds.insert(v4);

    return bounds;
}

template <typename T>
void split_primitive(aabb& L, aabb& R, float plane, int axis, QUAD_NS::swoop_quad<T> const& prim)
{
    auto v1 = prim.get_v1();
    auto v2 = prim.get_v2();
    auto v3 = prim.get_v3();
    auto v4 = prim.get_v4();

    L.invalidate();
    R.invalidate();

    detail::split_edge(L, R, v1, v2, plane, axis);
    detail::split_edge(L, R, v2, v4, plane, axis);
    detail::split_edge(L, R, v4, v3, plane, axis);
    detail::split_edge(L, R, v3, v1, plane, axis);
}

template <typename HR, typename T>
VSNRAY_FUNC
inline vector<3, T> get_normal(HR const& hr, QUAD_NS::swoop_quad<T> const& quad)
{
    VSNRAY_UNUSED(hr);

    return quad.normal;
}

// template <typename Normals, typename HR, typename T>
// VSNRAY_FUNC
// inline auto get_normal(
//         Normals                     normals,
//         HR const&                   hr,
//         QUAD_NS::swoop_quad<T>       prim,
//         per_vertex_binding          #<{(| |)}>#
//         )
//     -> decltype( get_normal(hr, prim) )
// {
//     VSNRAY_UNUSED(normals);
//
//     return get_normal(hr, prim);
// }
//
// template <typename Normals, typename HR, typename T>
// VSNRAY_FUNC
// inline auto get_shading_normal(
//         Normals                     normals,
//         HR const&                   hr,
//         QUAD_NS::swoop_quad<T>       #<{(| |)}>#,
//         per_vertex_binding          #<{(| |)}>#
//         )
//     -> typename std::iterator_traits<Normals>::value_type
// {
//     auto v1 = normals[hr.prim_id * 4];
//     auto v2 = normals[hr.prim_id * 4 + 1];
//     auto v3 = normals[hr.prim_id * 4 + 2];
//     auto v4 = normals[hr.prim_id * 4 + 3];
//
// //    return (1-hr.u) * (1-hr.v) * v1
// //         + hr.u     * (1-hr.v) * v2
// //         + hr.u     * hr.v     * v3
// //         + (1-hr.u) * hr.v     * v4;
//
//     auto a = slerp(v1, v2, v1, v2, hr.u);
//     auto b = slerp(v4, v3, v4, v3, hr.u);
//
//     auto r = slerp(a, b, a, b, hr.v);
//
//     return r;
// }


template <typename T>
struct num_vertices<QUAD_NS::swoop_quad<T>>
{
    enum { value = 4 };
};
template <typename T>
struct num_normals<QUAD_NS::swoop_quad<T>, per_face_binding>
{
    enum { value = 1 };
};
template <typename T>
struct num_normals<QUAD_NS::swoop_quad<T>, per_vertex_binding>
{
    enum { value = 4 };
};
template <typename T>
struct num_tex_coords<QUAD_NS::swoop_quad<T>>
{
    enum { value = 4 };
};

} // visionaray

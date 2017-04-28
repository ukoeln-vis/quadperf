namespace QUAD_NS
{

namespace detail {

//-------------------------------------------------------------------------------------------------
// intersect by using test with Pluecker coordinates (cf. Shevtsov et al. 2007,
//      "Ray-Triangle Intersection Algorithm for Modern CPU Architectures"
// TODO: implement/test the precalculations proposed in the paper
//

template <typename R>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> intersect_pluecker(R const& ray, basic_quad<float> const& quad)
{
    using T  = typename R::scalar_type;
    using V  = vector<3, T>;
    using V2 = vector<2, T>;

    hit_record<R, primitive<unsigned>> result;

    // vector<6, T> e1(quad.v1 - quad.v2, cross(quad.v1, quad.v2));
    // vector<6, T> e2(quad.v2 - quad.v3, cross(quad.v2, quad.v3));
    // vector<6, T> e3(quad.v3 - quad.v4, cross(quad.v3, quad.v4));
    // vector<6, T> e4(quad.v4 - quad.v1, cross(quad.v4, quad.v1));
    // vector<6, T> r(cross(ray.dir, ray.ori), ray.dir);

    // load only once
    V quad_v1(quad.v1);
    V quad_v2(quad.v2);
    V quad_v3(quad.v3);
    V quad_v4(quad.v4);

    vector<3, T> e1(cross(quad_v1 - quad_v2, quad_v2 - ray.ori));
    vector<3, T> e2(cross(quad_v2 - quad_v3, quad_v3 - ray.ori));
    vector<3, T> e3(cross(quad_v3 - quad_v4, quad_v4 - ray.ori));
    vector<3, T> e4(cross(quad_v4 - quad_v1, quad_v1 - ray.ori));
    vector<3, T> r(ray.dir);


    T s1 = copysign(T(1.0), dot(e1, r));
    T s2 = copysign(T(1.0), dot(e2, r));
    T s3 = copysign(T(1.0), dot(e3, r));
    T s4 = copysign(T(1.0), dot(e4, r));

    result.hit = s1 == s2 && s1 == s3 && s1 == s4;

    if (any(result.hit))
    {
        V v1(quad_v1);
        V e1(quad_v2 - quad_v1);
        V e2(quad_v3 - quad_v2);

        V n = normalize(cross(e1, e2));

        T div = dot(n, ray.dir);

        // ray/plane intersection
        result.t = select(
                div != T(0.0),
                dot(v1 - ray.ori, n) / div,
                result.t
                );
        // result.prim_id = quad.prim_id;
        // result.geom_id = quad.geom_id;
        result.isect_pos = ray.ori + ray.dir * result.t;
#if CALCULATE_UV
        V2 uv = get_uv(quad, result.isect_pos);
        result.u = uv.x;
        result.v = uv.y;
#endif
    }

    return result;
}

} // detail

} // QUAD_NS

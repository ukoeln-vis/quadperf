namespace QUAD_NS
{

namespace detail {

//-------------------------------------------------------------------------------------------------
// intersect by unconditionally calculating uv and checking if they are in [0..1]
//      "Schlick and Subrenat 1995 (?)"
//

template <typename R>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> intersect_uv(R const& ray, basic_quad<float> const& quad)
{
    using T  = typename R::scalar_type;
    using V  = vector<3, T>;
    using V2 = vector<2, T>;

    hit_record<R, primitive<unsigned>> result;

    V v1(quad.v1);
    V e1(quad.v2 - quad.v1);
    V e2(quad.v3 - quad.v2);

    V n = normalize(cross(e1, e2));

    T div = dot(n, ray.dir);

    // ray/plane intersection
    T t = select(
            div != T(0.0),
            dot(v1 - ray.ori, n) / div,
            T(-1.0)
            );

    result.hit = div != T(0.0) && t >= T(0.0);
    result.t = t;

    if (!any(result.hit))
    {
        return result;
    }

    V isect_pos = ray.ori + ray.dir * t;


    V2 uv = get_uv(quad, isect_pos);

    result.hit = uv.x >= T(0.0) && uv.x <= T(1.0) && uv.y >= T(0.0) && uv.y <= T(1.0);

    //result.hit = uv.x >= T(0.) && uv.x <= T(1.) && uv.y >= T(0.) && uv.y <= T(0.99999);

    // result.prim_id = quad.prim_id;
    // result.geom_id = quad.geom_id;
    result.isect_pos = isect_pos;
    result.u = uv.x;
    result.v = uv.y;

    return result;
}

} // detail

} // QUAD_NS

/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2011-2022 Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <nanoflann.hpp>

#include "utils.h"

template <typename num_t>
void kdtree_demo(const size_t N)
{
    PointCloud<num_t> cloud;

    // construct a kd-tree index:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>,
        PointCloud<num_t>, 3 /* dim */
        >;

    my_kd_tree_t index1(3 /*dim*/, cloud, {10 /* max leaf */});
    my_kd_tree_t index2(3 /*dim*/, cloud, {10 /* max leaf */});

    // Generate points:
    generateRandomPointCloud(cloud, N);

    num_t query_pt[3] = {0.5, 0.5, 0.5};
    cloud.pts[N - 1] = {query_pt[0], query_pt[1], query_pt[2]};

    auto start = std::chrono::steady_clock::now();

    // add points in chunks at a time
    size_t chunk_size = 100;
    for (size_t i = 0; i < N; i = i + chunk_size)
    {
        size_t end = std::min<size_t>(i + chunk_size, N - 1);
        // Inserts all points from [i, end]
        index1.addPointsOld(i, end);
    }

    // remove a point
    size_t removePointIndex = N - 1;
    index1.removePoint(removePointIndex);
    index1.addPointsOld(N - 1, N - 1);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Old " << diff.count() << "sec" << std::endl;

    start = std::chrono::steady_clock::now();
    // add points in chunks at a time
    for (size_t i = 0; i < N; i = i + chunk_size)
    {
        size_t end = std::min<size_t>(i + chunk_size, N - 1);
        // Inserts all points from [i, end]
        index2.addPoints(i, end);
    }

    // remove a point
    index2.removePoint(removePointIndex);
    index2.addPoints(removePointIndex, removePointIndex);

    end = std::chrono::steady_clock::now();
    diff = end - start;

    std::cout << "New " << diff.count() << "sec" << std::endl;

    {
        std::cout << "Searching for 1 element..." << std::endl;
        // do a knn search
        const size_t                   num_results = 1;
        size_t                         ret_index;
        num_t                          out_dist_sqr;
        nanoflann::KNNResultSet<num_t> resultSet1(num_results), resultSet2(num_results);
        resultSet1.init(&ret_index, &out_dist_sqr);
        index1.findNeighbors(resultSet1, query_pt, {10});
        std::cout << "point: (" << cloud.pts[ret_index].x << ", "
                  << cloud.pts[ret_index].y << ", " << cloud.pts[ret_index].z
                  << ")" << std::endl;
        resultSet2.init(&ret_index, &out_dist_sqr);
        index2.findNeighbors(resultSet2, query_pt, {10});
        std::cout << "point: (" << cloud.pts[ret_index].x << ", "
                  << cloud.pts[ret_index].y << ", " << cloud.pts[ret_index].z
                  << ")" << std::endl;
    }
    {
        // do a knn search searching for more than one result
        const size_t num_results = 5;
        std::cout << "Searching for " << num_results << " elements"
                  << std::endl;
        size_t                         ret_index[num_results];
        num_t                          out_dist_sqr[num_results];
        nanoflann::KNNResultSet<num_t> resultSet1(num_results), resultSet2(num_results);
        resultSet1.init(ret_index, out_dist_sqr);
        resultSet2.init(ret_index, out_dist_sqr);
        index1.findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));
    }
    {
        // Unsorted radius search:
        std::cout << "Unsorted radius search" << std::endl;
        const num_t                               radius = 1;
        std::vector<std::pair<size_t, num_t>>     indices_dists;
        nanoflann::RadiusResultSet<num_t, size_t> resultSet1(
            radius, indices_dists), resultSet2( radius, indices_dists);
        index1.findNeighbors(resultSet1, query_pt, nanoflann::SearchParams());
    }
}

int main()
{
    // Randomize Seed
    srand(static_cast<unsigned int>(time(nullptr)));
    for(int i = 0; i < 10; i++)
        kdtree_demo<float>(1000000);
    std::cout << "=======" << std::endl;
    for(int i = 0; i < 10; i++)
        kdtree_demo<double>(1000000);
    return 0;
}

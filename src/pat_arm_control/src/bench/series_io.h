#pragma once
/*! The per-tick record read back from disk: the inverse of writeSeriesCsv().
 *
 *  A run recorded by anything other than the bench (the live stack's recorder
 *  node writes the same columns) is scored by reading its CSV here and handing
 *  the result to summarise(), so a live RMSE is computed by exactly the code that
 *  computes a bench RMSE. Columns this reader does not know are ignored, so a
 *  newer writer does not break an older tool.
 */
#include <string>

#include "run_summary.h"

namespace pat_arm_control {
namespace bench {

/*! Read a series CSV. Throws std::runtime_error if the file cannot be read, a
 *  required column is missing, or a row has the wrong number of fields. The
 *  per-joint limits (`tau_limit`, `q_min`, `q_max`) are not in the file; the
 *  caller sets them on the result. */
RunSeries readSeriesCsv(const std::string& path);

}  // namespace bench
}  // namespace pat_arm_control

// NaiveBayes.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>


using PaperID = int64_t;
using PaperTitle = std::string;
using Point = std::pair<PaperID, PaperTitle>;

std::vector<Point> consume_metadata(const std::string& filename)
{
	std::vector<Point> points;

	std::ifstream metadata{ filename };


	//Read in the "reduced" title with no punctuation, all lowercase, etc


	PaperTitle paper_orig_title;
	PaperTitle paper_title;
	std::string paper_id_raw;
	while (std::getline(metadata, paper_id_raw, '\t'))
	{
		std::getline(metadata, paper_orig_title, '\t');
		std::getline(metadata, paper_title, '\t');

		PaperID paper_id = std::stoll(paper_id_raw, nullptr, 16);

		points.emplace_back(paper_id, paper_title);
		metadata.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

	return points;
}

int __cdecl main(int argc, char* argv[])
{
	consume_metadata("metadata/icse_id.txt");
	consume_metadata("metadata/vldb_id.txt");
    return 0;
}


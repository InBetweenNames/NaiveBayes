// NaiveBayes.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


using PaperID = int64_t;
using Point = std::pair<PaperID, std::string>;
using ClassMetadata = std::pair<std::string, std::vector<Point>>;

std::vector<Point> consume_metadata(const std::string& filename)
{
	std::vector<Point> points;

	std::ifstream metadata{ filename };


	//Read in the "reduced" title with no punctuation, all lowercase, etc


	std::string paper_orig_title;
	std::string paper_title;
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
	std::vector<std::pair<std::string, std::string>> metadata_files;
	int n_features = 0;
	if (argc < 2)
	{
		metadata_files.emplace_back("icse", "metadata/icse_id.txt");
		metadata_files.emplace_back("vldb", "metadata/vldb_id.txt");
	}
	else
	{
		std::string first_arg{ argv[1] };
		int start_index = 1;
		if (first_arg == "--selectfeatures")
		{
			if (argc < 3)
			{
				n_features = 100; //Default to selecting 100 features
				start_index = 2;
			}
			else
			{
				try
				{
					n_features = std::stoi(argv[2]);
					start_index = 3;
					if (n_features < 0)
					{
						n_features = 100;
					}
				}
				catch (const std::invalid_argument& ex)
				{
					start_index = 2;
				}
			}
		}
		if (first_arg == "--help" || (argc - start_index) % 2 != 0)
		{
			std::cout << "Usage: " << argv[0] << " (--selectfeatures (n)) [class1 class1filename class2 class2filename ... classN classNfilename]" << std::endl;
			std::cout << "If no arguments are provided, this command line will be run: " << argv[0] << " icse metadata/icse_id.txt vldb metadata/vldb_id.txt" << std::endl;

			return 0;
		}
		for (int i = start_index; i < argc; i += 2)
		{
			const auto* class_name = argv[i];
			const auto* filename = argv[i + 1];
			metadata_files.emplace_back(class_name, filename);
		}

	}

	std::vector<ClassMetadata> metadata;

	for (const auto& file : metadata_files)
	{
		const auto& class_metadata = consume_metadata(file.second);
		metadata.emplace_back(file.first, class_metadata);
	}

    return 0;
}


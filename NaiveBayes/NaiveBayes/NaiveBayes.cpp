// NaiveBayes.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <fstream>
#include <set>
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

//Extract all words from a vector of metadata and add them to a vocabulary set
void extract_vocabulary(const std::vector<Point>& metadata, std::set<std::string>& vocabulary)
{
	for (const auto& docs : metadata)
	{
		const auto& doc = docs.second;
		std::stringstream tokens{ doc };
		std::string token;
		while (tokens >> token)
		{
			vocabulary.emplace(token);
		}
	}
}

/*
Eigen::VectorXi count_occurrences(const std::vector<Point>& metadata, const std::set<std::string>& vocabulary)
{

}
*/

class NaiveBayes
{
	int64_t N = 0;
	std::vector<int64_t> n_docs_in_class;
public:

	//Takes a vector of pairs of iterators as a data source
	template <typename T>
	NaiveBayes(const std::vector<std::pair<T, T>>& metadata)
	{
		/*for (size_t i = 0; i < metadata; i++)
		{

		}*/
		for (const auto C : metadata)
		{
			const auto Nc = std::distance(C.first, C.second);
			n_docs_in_class.emplace_back(Nc);
			N += Nc;
		}

	}
};

Eigen::Array<Eigen::Array2i, -1, -1> get_m_fold_slices(const std::vector<ClassMetadata>& metadata, int M)
{
	const auto n_classes = metadata.size();
	Eigen::Array<Eigen::Array2i, -1, -1> indices{M, n_classes};

	for (int i = 0; i < n_classes; i++)
	{
		const auto Nc = metadata[i].second.size();
		const auto n_per_fold = Nc / M;
		//const auto n_per_fold_r = Nc % M;

		int curr = 0;
		for (int j = 0; j < M; j++)
		{
			const auto start = j*n_per_fold;


			indices(j, i)(0) = start;
			
			//Take care to include remainder in the last fold
			if (j == M - 1)
			{
				const auto end = Nc;
				indices(j, i)(1) = end;
			}
			else
			{
				const auto end = (j + 1)*n_per_fold;
				indices(j, i)(1) = end;
			}
		}
	}

	return indices;
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
				catch (...)
				{
					start_index = 2;
				}
			}
		}
		if (first_arg == "--help" || (argc - start_index) % 2 != 0)
		{
			std::cout << "Usage: " << argv[0] << " (--selectfeatures (<n>)) [class1 class1filename class2 class2filename ... classN classNfilename]" << std::endl;
			std::cout << "If no arguments are provided, this command line will be run:\n\t" << argv[0] << " --selectfeatures 100 icse metadata/icse_id.txt vldb metadata/vldb_id.txt" << std::endl;

			return 0;
		}
		for (int i = start_index; i < argc; i += 2)
		{
			const auto* class_name = argv[i];
			const auto* filename = argv[i + 1];
			metadata_files.emplace_back(class_name, filename);
		}

	}

	//std::vector<ClassMetadata> metadata;

	std::vector<ClassMetadata> metadata;

	for (const auto& file : metadata_files)
	{
		const auto& class_metadata = consume_metadata(file.second);
		metadata.emplace_back(file.first, class_metadata);
	}

	const auto folds = get_m_fold_slices(metadata, 10);

	//std::vector<std::vector<std::pair<it, it>>> = mFoldCrossValidate(metadata, 10);

    return 0;
}


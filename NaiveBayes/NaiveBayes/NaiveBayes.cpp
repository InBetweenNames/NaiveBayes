// NaiveBayes.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <Eigen/StdVector>
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

template <typename T>
using Vec = std::vector<T, Eigen::aligned_allocator<T>>;

Vec<Vec<Eigen::Array2i>> train_all(const std::vector<ClassMetadata>& metadata)
{
	Vec<Vec<Eigen::Array2i>> classes;
	for (const auto& C : metadata)
	{
		Eigen::Array2i indices{ 0, static_cast<int>(C.second.size()) };
		classes.emplace_back(Vec<Eigen::Array2i>{indices});
	}

	return classes;
}

class NaiveBayes
{
	int64_t N = 0;
	std::vector<int64_t> n_docs_in_class;
public:


	template <typename T>
	NaiveBayes(const std::vector<ClassMetadata>& metadata, Vec<Vec<T>> trainingIndices)
	{
		/*for (size_t i = 0; i < metadata; i++)
		{

		}*/
		const auto n_classes = metadata.size();

		for (size_t i = 0; i < n_classes; i++)
		{
			const auto& classIndices = trainingIndices[i];
			Eigen::Index Nc = 0;
			for (const auto& indices : classIndices)
			{
				const auto start = indices(0);
				const auto end = indices(1);
				Nc += end - start;
			}
			n_docs_in_class.emplace_back(Nc);
			N += Nc;
		}

	}

	//Trains using entire metadata set
	NaiveBayes(const std::vector<ClassMetadata>& metadata) : NaiveBayes{ metadata, train_all(metadata) }
	{

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

		for (int j = 0; j < M; j++)
		{
			const auto start = j*n_per_fold;


			indices(j, i)(0) = static_cast<int>(start);
			
			//Take care to include remainder in the last fold
			if (j == M - 1)
			{
				const auto end = Nc;
				indices(j, i)(1) = static_cast<int>(end);
			}
			else
			{
				const auto end = (j + 1)*n_per_fold;
				indices(j, i)(1) = static_cast<int>(end);
			}
		}
	}

	return indices;
}

template <typename T>
Vec<Eigen::Array2i> get_m_fold_training_indices(const T& indices, const Eigen::Index exclude)
{
	Vec<Eigen::Array2i> chunks;

	if (exclude != 0)
	{
		Vector2i chunk = { indices(0)(0), indices(exclude - 1)(1) };
		chunks.emplace_back(chunk);
	}

	if (exclude != indices.rows() - 1)
	{
		Vector2i chunk = { indices(exclude + 1)(0), indices(indices.rows() - 1)(1) };
		chunks.emplace_back(chunk);
	}

	return chunks;
}

template <typename T>
Eigen::Array2i get_m_fold_testing_indices(const T& indices, const Eigen::Index include)
{
	return indices.row(include);
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

	NaiveBayes all{ metadata };

	//std::vector<std::vector<std::pair<it, it>>> = mFoldCrossValidate(metadata, 10);

    return 0;
}

